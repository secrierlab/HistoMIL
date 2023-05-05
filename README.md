# HistoMIL
![alt text](https://github.com/RunningStone/HistoMIL/raw/main/logo.png)


HistoMIL is a Python package for handling histopathology whole-slide images using multiple instance learning (MIL) techniques. With HistoMIL, you can create MIL datasets, train and evaluate MIL models, and make MIL predictions on new slide images.

## Getting Started

To use HistoMIL, you first need to create a conda environment with the required dependencies. You can do this by importing the env.yml file provided in this repository:

```bash
conda env create -f env.yml
```
This will create a new environment named histomil, which you can activate with:

```bash
conda activate histomil
```

Next, install the required Python packages with pip:

```bash
pip install -r requirements.txt
```
This will install all the packages listed in requirements.txt, including HistoMIL itself.

## Usage

All of the examples for using HistoMIL are included in the Notebooks folder. You can open and run these Jupyter notebooks to see how to use HistoMIL for different histopathology tasks.

## Contributing

If you find a bug or want to suggest a new feature for HistoMIL, please open a GitHub issue in this repository. Pull requests are also welcome!

## License

HistoMIL is released under the MIT License. See the LICENSE file for more information.