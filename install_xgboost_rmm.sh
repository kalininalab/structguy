#!/bin/bash
#
# Builds and installs the newest release of xgboost from source with the
# RMM (RAPIDS Memory Manager) plugin enabled, i.e. configured with
# -DUSE_CUDA=ON -DPLUGIN_RMM=ON. The PyPI xgboost wheels are never built
# with this plugin, so structguy's `use_rmm=True` / `use_cuda_async_pool=True`
# config contexts (support_classes.py, trainForest.py) silently do nothing
# unless xgboost is built this way locally.
#
# Individually callable, e.g.:
#   ./install_xgboost_rmm.sh -e structman_minnie
#
# Requires a conda env that already has a CUDA toolkit (>=12.9, matching
# upstream xgboost's current minimum) and librmm installed, e.g. via:
#   mamba install -c rapidsai -c conda-forge -c nvidia "cuda-toolkit>=12.9" librmm

# Absolute path to this script.
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in.
SCRIPTPATH=$(dirname "$SCRIPT")

#Init default arguments
env_name=""
cuda_archs="native"
verbose=false

#Parse arguments
while getopts e:a:v flag
do
    case "${flag}" in
        e) env_name=${OPTARG};;
        a) cuda_archs=${OPTARG};;
        v) verbose=true;;
    esac
done

verbose_stdout=3
verbose_stderr=4

if [ "$verbose" = true ]; then
    eval "exec $verbose_stdout>&1"
    eval "exec $verbose_stderr>&2"
else
    eval "exec $verbose_stdout>/dev/null"
    eval "exec $verbose_stderr>/dev/null"
fi

# Runs a command, always logging its output to a file. In verbose mode the
# output also streams live; either way, on failure the tail of the log is
# dumped to real stdout so the actual error is never hidden behind -v.
run_logged() {
    local log_file="$1"
    shift
    "$@" > >(tee "$log_file" >&$verbose_stdout) 2> >(tee -a "$log_file" >&$verbose_stderr)
    local status=$?
    if [ $status -ne 0 ]; then
        echo "---- command failed (exit $status): $* ----"
        echo "---- last 60 lines of $log_file ----"
        tail -n 60 "$log_file"
    fi
    return $status
}

#Activate the target conda environment (unless one is already active)
if [ -n "$env_name" ]
then
    conda_base_path=$(conda info --base)
    conda_bash_path="$conda_base_path"/etc/profile.d/conda.sh
    source "$conda_bash_path"
    conda activate "$env_name" >&$verbose_stdout 2>&$verbose_stderr
    if [ $? -ne 0 ]
    then
        echo "Failed to activate conda environment '$env_name'"
        exit 1
    fi
fi

if [ -z "$CONDA_PREFIX" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]
then
    echo "No non-base conda environment is active."
    echo "Either activate one first, or call this script with -e <env_name>."
    exit 1
fi

echo "Building xgboost with RMM support in conda environment '$CONDA_DEFAULT_ENV' ..."

build_temp_folder=$(mktemp -d -t xgboost_rmm_build.XXXXXXX)
trap 'rm -rf -- "$build_temp_folder"' EXIT

#Make sure a CUDA toolkit and librmm are present in the environment
mamba_version_test_output=$(mamba --version 2>/dev/null)
if [ -z "$mamba_version_test_output" ]
then
    run_logged "$build_temp_folder/mamba_bootstrap.log" conda install -y -c conda-forge mamba
fi

run_logged "$build_temp_folder/deps_install.log" \
    mamba install -y -c rapidsai -c conda-forge -c nvidia "cuda-toolkit>=12.9" librmm
if [ $? -ne 0 ]
then
    echo "Installing cuda-toolkit/librmm failed, aborting."
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1
then
    echo "nvcc not found on PATH after installing cuda-toolkit, aborting."
    exit 1
fi

if [ ! -d "$CONDA_PREFIX/include/rmm" ]
then
    echo "librmm headers not found under \$CONDA_PREFIX/include/rmm, aborting."
    exit 1
fi

#librmm's CMake config requires a much newer CMake than what's usually on
#PATH (system cmake or an older conda one); bootstrap a fresh one via pip so
#it lives in $CONDA_PREFIX/bin and takes precedence once the env is active.
required_cmake_version="3.30.4"
run_logged "$build_temp_folder/cmake_bootstrap.log" pip install "cmake>=3.30"
if [ $? -ne 0 ]
then
    echo "Installing a recent cmake via pip failed, aborting."
    exit 1
fi

installed_cmake_version=$(cmake --version | head -1 | awk '{print $3}')
if [ "$(printf '%s\n%s\n' "$required_cmake_version" "$installed_cmake_version" | sort -V | head -1)" != "$required_cmake_version" ]
then
    echo "cmake on PATH is still $installed_cmake_version (from $(command -v cmake)), need >= $required_cmake_version."
    echo "Check that \$CONDA_PREFIX/bin precedes other cmake installs on PATH."
    exit 1
fi

#Determine the newest stable xgboost release tag
echo "Looking up newest xgboost release tag ..."
xgboost_tag=$(git ls-remote --tags --refs https://github.com/dmlc/xgboost.git \
    | awk -F'refs/tags/' '{print $2}' \
    | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+(-post[0-9]+)?$' \
    | sort -V \
    | tail -1)

if [ -z "$xgboost_tag" ]
then
    echo "Could not determine the newest xgboost release tag from GitHub, aborting."
    exit 1
fi
echo "Building xgboost $xgboost_tag from source ..."

run_logged "$build_temp_folder/clone.log" \
    git clone --branch "$xgboost_tag" --depth 1 --recurse-submodules \
        https://github.com/dmlc/xgboost.git "$build_temp_folder/xgboost"

if [ ! -f "$build_temp_folder/xgboost/CMakeLists.txt" ]
then
    echo "Failed to clone xgboost $xgboost_tag, aborting."
    exit 1
fi

#CMAKE_CUDA_ARCHITECTURES=native needs an actual GPU visible at configure
#time to query its compute capability; decide up front instead of guessing
#from a failed cmake run, so a real configure error never gets misreported
#as a missing GPU.
if [ "$cuda_archs" = "native" ] && ! (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "GPU")
then
    echo "No GPU visible on this build host, cannot use CMAKE_CUDA_ARCHITECTURES=native."
    echo "Falling back to a fixed set of common architectures (75;80;86;89;90)."
    echo "Pass -a <archs> to target something else, e.g. if building on a login node for a different GPU generation."
    cuda_archs="75;80;86;89;90"
fi

#xgboost's RMM plugin source is only tested by upstream against the
#not-yet-released next RMM version (see ops/pipeline/nightly-test-rmm*.sh
#upstream), which relocated rmm/mr/device/thrust_allocator_adaptor.hpp to
#rmm/mr/thrust_allocator_adaptor.hpp. Forcing the matching librmm here would
#require upgrading the whole RAPIDS stack already pinned in this env (cudf,
#cuxfilter, cuspatial, ... all lock rmm to their own X.Y band), so instead
#shim the old, still-installed header path to the new one xgboost expects.
#The class itself (rmm::mr::thrust_allocator) is unchanged, only its path.
rmm_compat_include="$build_temp_folder/rmm_compat_include"
mkdir -p "$rmm_compat_include/rmm/mr"
if [ ! -f "$CONDA_PREFIX/include/rmm/mr/thrust_allocator_adaptor.hpp" ] \
    && [ -f "$CONDA_PREFIX/include/rmm/mr/device/thrust_allocator_adaptor.hpp" ]
then
    echo "Shimming rmm/mr/thrust_allocator_adaptor.hpp -> rmm/mr/device/thrust_allocator_adaptor.hpp (installed librmm is older than what xgboost's source expects) ..."
    cat > "$rmm_compat_include/rmm/mr/thrust_allocator_adaptor.hpp" <<'EOF'
#pragma once
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>
EOF
fi

pushd "$build_temp_folder/xgboost" >/dev/null

echo "Configuring xgboost build (USE_CUDA=ON, PLUGIN_RMM=ON, CMAKE_CUDA_ARCHITECTURES=$cuda_archs) ..."
run_logged "$build_temp_folder/configure.log" \
    cmake -B build -S . \
        -DUSE_CUDA=ON \
        -DPLUGIN_RMM=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$cuda_archs" \
        -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
        -DCMAKE_CUDA_FLAGS="-I$rmm_compat_include" \
        -DCMAKE_CXX_FLAGS="-I$rmm_compat_include" \
        -DCMAKE_BUILD_TYPE=Release
if [ $? -ne 0 ]
then
    echo "cmake configuration of xgboost failed, aborting."
    popd >/dev/null
    exit 1
fi

echo "Compiling xgboost (this can take a while) ..."
run_logged "$build_temp_folder/build.log" \
    cmake --build build --target xgboost -j"$(nproc)"
if [ $? -ne 0 ] || [ ! -f "lib/libxgboost.so" ]
then
    echo "xgboost build failed (or did not produce lib/libxgboost.so), aborting."
    popd >/dev/null
    exit 1
fi

echo "Installing the xgboost python package (bundling the RMM-enabled build) ..."
run_logged "$build_temp_folder/pip_install.log" pip install -v ./python-package
install_status=$?

popd >/dev/null

if [ $install_status -ne 0 ]
then
    echo "pip install of the xgboost python package failed, aborting."
    exit 1
fi

echo "Verifying installation ..."
python -c "
import xgboost
print('xgboost version:', xgboost.__version__)
try:
    print('build_info:', xgboost.build_info())
except Exception as e:
    print('Could not query xgboost.build_info():', e)
"

echo "xgboost $xgboost_tag with RMM support successfully installed into '$CONDA_DEFAULT_ENV'."
exit 0
