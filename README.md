# HistoMIL
![HistoMIL](https://github.com/RunningStone/HistoMIL/raw/main/logo_full.png)


HistoMIL is a Python package for handling histopathology whole-slide images using multiple instance learning (MIL) techniques. With HistoMIL, you can create MIL datasets, train and evaluate MIL models, and make MIL predictions on new slide images.

## Getting Started

To use HistoMIL, you first need to create a conda environment with the required dependencies. You can do this by importing the env.yml file provided in this repository:

```bash
conda create -n HistoMIL python=3.9
```
This will create a new environment named histomil, which you can activate with:

```bash
conda activate HistoMIL
```

Then install openslide and pytorch-gpu with following scripts.

```bash
conda install -c conda-forge openslide
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
```bash

Next, install the required Python packages with pip:

```bash
pip install pytorch-lightning=1.9
pip install pandas openslide-python torchvision imageio matplotlib timm scikit-learn opencv-python h5py
pip install wandb
```
This will install all the packages listed in requirements.txt, including HistoMIL itself.

## Usage

All of the examples for using HistoMIL are included in the Notebooks folder. You can open and run these Jupyter notebooks to see how to use HistoMIL for different histopathology tasks.

## Contributing

If you find a bug or want to suggest a new feature for HistoMIL, please open a GitHub issue in this repository. Pull requests are also welcome!

## License

HistoMIL is released under the GNU-GPL License. See the LICENSE file for more information.
