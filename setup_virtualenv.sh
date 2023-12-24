#!/bin/bash

virtualenv --python=python3.7 .env
source .env/bin/activate

pip install -U pip
pip install -U setuptools wheel twine
echo 'Installing DoXpy'
pip install -e expert_vs_gpt_vs_doxpy/code/packages/doxpy
pip install -e expert_vs_gpt_vs_doxpy/code/packages/quansx

python -m spacy download en_core_web_trf
python -m spacy download en_core_web_md
python -m nltk.downloader stopwords punkt averaged_perceptron_tagger framenet_v17 wordnet brown omw-1.4
