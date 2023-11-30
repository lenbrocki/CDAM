# Class-Discriminative Attention Maps (CDAM)

![Alt text](readme-images/sample.png?raw=true "Title")

CDAM is a novel post-hoc explanation method for vision transformers (ViTs) that is highly sensitive to the chosen target class and reveals evidence and counter-evidence through signed relevance scores.

## Run notebook locally

To run [CDAM.ipynb](CDAM.ipynb) locally we recommend to create a Python >= 3.9 virtual environment, for example with [Mamba](https://github.com/mamba-org/mamba). Inside the environment run ```pip install -r requirements_local.txt``` and create a Jupyter kernel with ```python -m ipykernel install --user --name cdam_kernel```.

## Run notebook on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lenbrocki/CDAM/blob/main/attention_map.ipynb)
