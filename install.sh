#!/bin/bash

# Absolute path to this script.
SCRIPT=$(readlink -f $0)
# Absolute path this script is in.
SCRIPTPATH=$(dirname "$SCRIPT")

#Init constants
current_python_version="3.12"
username=$(whoami)

#Init default arguments
env_name=""
storage_folder=""

verbose=false
overwrite=false
rewrite=false

installer_temp_folder=$(mktemp -d -t structguy.XXXXXXX)
trap 'rm -rf -- "$installer_temp_folder"' EXIT


#Parse arguments
while getopts s:e:vor flag
do
    case "${flag}" in
        e) env_name=${OPTARG};;
        s) storage_folder=${OPTARG};;
        v) verbose=true;;
        o) overwrite=true;;
        r) rewrite=true;;
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


#Check if conda environment already exits, create it if not
env_list_result=$(conda env list | grep -w "$env_name")
if [ -z "$env_list_result" ]
then
    echo "Conda environment with name $env_name not in current environment list, please provide a valid environment including a StructMAn installation"
    exit 1
fi

#Locate conda source to activate conda inside bash script shell
conda_base_path=$(conda info --base)
conda_bash_path="$conda_base_path"/etc/profile.d/conda.sh

current_env=$(conda info | grep 'active environment' | awk '{sub(/^[ \t\r\n]+active environment.:./,""); print}')
if ! [ "$current_env" = "base" ]
then
    echo "Conda needs to be in base environment, but is $current_env"
    echo "Please call 'conda deactivate'."
    echo "exiting installer ..."
    exit 1
fi

#activate the environment
{
    echo "Activating environment inside shell ..."
    source "$conda_bash_path"
    current_env=$(conda info | grep 'active environment' | awk '{sub(/active environment.:./,""); print}')

    if ! [ "$current_env" = "base" ]
    then
        conda deactivate
    fi

    new_env_path=$(conda env list | awk -v name="$env_name" '/^[^#]/{ if ($1 == name) {print $2} }')
    conda activate "$env_name"
    echo "$env_name" activated
} >&$verbose_stdout


conda_list_result=$(conda list | grep structman)
if [ -z "$env_list_result" ]
then
    echo "StructMAn seems not be installed in $env_name environment, please provide a valid environment including a StructMAn installation"
    exit 1
fi

mamba_version_test_output=$(mamba --version 2>/dev/null)

if [ -z "$mamba_version_test_output" ]
then
    conda install -y -c conda-forge mamba >&$verbose_stdout 2>&$verbose_stderr
fi

#install dependencies
{
    mamba install -y -c conda-forge scip==9.0.0
    echo "Installing package DataSAIL ..."
    mamba install -y -c conda-forge -c kalininalab -c bioconda -c mosek datasail==1.2.1
    pip install grakel
} >&$verbose_stdout

#install the main package
echo "Installing StructGuy source code using pip ..."
pip install "$SCRIPTPATH" >&$verbose_stdout

if [ -z $storage_folder ]
then
    storage_folder="$new_env_path"/share/structguy
    if ! [ -d "$storage_folder" ]
    then
        mkdir "$storage_folder"
    fi
fi

if [ -d $storage_folder ]
then
    storage_folder=$(realpath "$storage_folder")
else
    mkdir "$storage_folder"
    storage_folder=$(realpath "$storage_folder")
fi


if ! [ -d "$storage_folder/$username" ]
then
    mkdir "$storage_folder/$username"
    echo "Needed to create a user-specific folder on scratch to locate the tmp folder:"
    echo "   $storage_folder/$username"
fi

tmp_folder_path="$storage_folder/$username/tmp"
if ! [ -d "$tmp_folder_path" ]
then
    mkdir "$tmp_folder_path"
    echo "Needed to create the tmp folder:"
    echo "   $tmp_folder_path"
fi

#install psic
psic_folder_path="$new_env_path"/lib/python"$current_python_version"/site-packages/structguy/resources/psic/
pushd $psic_folder_path
make psic
popd

#Download and construct the indices for the search databases UniRef50 and UniRef90
#resources_folder_path="$new_env_path"/lib/python"$current_python_version"/site-packages/structguy/resources/
resources_folder_path="$new_env_path"/lib/python"$current_python_version"/site-packages/
pushd $storage_folder
if [ "$overwrite" = true ]; then
    rm -rf *.gz
fi

do_ref50=false
if [ "$rewrite" = true ]
then
    do_ref50=true
fi
if ! [ -f uniref50.fasta.gz ] 
then
    do_ref50=true
fi

if [ $do_ref50 = true ]
then
    if ! [ -f uniref50.fasta.gz ]
    then
        wget ftp://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz
    fi
    split_fasta_db uniref50.fasta.gz "$tmp_folder_path"
    #mmseqs createdb uniref50.fasta.gz uniref50_search_db
    #mmseqs createindex uniref50_search_db "$tmp_folder_path" -s 7.5
fi

do_ref90=false
if [ "$rewrite" = true ]
then
    do_ref90=true
fi
if ! [ -f uniref90.fasta.gz ] 
then
    do_ref90=true
fi

if [ $do_ref90 = true ]
then
    if ! [ -f uniref90.fasta.gz ]
    then
        wget ftp://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz
    fi
    split_fasta_db uniref90.fasta.gz "$tmp_folder_path"
    #mmseqs createdb uniref90.fasta.gz uniref90_search_db
    #mmseqs createindex uniref90_search_db "$tmp_folder_path" -s 7.5
fi

echo "search_db_folder=$storage_folder" > "$resources_folder_path"search_db_settings.conf
popd

echo "StructGuy successfully installed, please activate the right conda environment before using it:"
echo "    conda activate $env_name"

exit 0