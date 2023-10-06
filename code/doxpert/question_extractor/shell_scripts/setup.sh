echo 'Create a virtual environment'
python3 -m venv .env
echo 'Activate the virtual environment'
source .env/bin/activate
echo 'Update the virtual environment'
pip install -U pip setuptools wheel utils
echo 'Install quansx'
pip install -e ../../../packages/quansx
pip install -r ../data/requirements.txt
