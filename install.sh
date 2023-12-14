cd structguy
git branch
cd ..

conda_base_path=$(conda env list | grep base | sed -e 's/.* \([^ ]*\)$/\1/g') # Regex fetches the base path to conda
conda_bash_path="$conda_base_path"/etc/profile.d/conda.sh
source "$conda_bash_path"

git clone https://github.com/kalininalab/structman_dev.git
git clone https://github.com/kalininalab/structguy.git

cd structman_dev

./install.sh -d dmias -p bZ5YwDBEPBPom06V -m MODELIRANJE -e structman

cd ..

conda activate structman

conda env list

cd structguy

git branch
sleep 10

pip install .

conda install -y -c conda-forge mamba

conda update -y --force conda
mamba install -y -c conda-forge -c kalininalab -c bioconda -c mosek DataSAIL
pip install grakel

mamba install -y -c conda-forge rdkit==2023.03.3
mamba install -y -c conda-forge scip==8.0.3
pip install scikit-learn==1.2.2

cd ..
