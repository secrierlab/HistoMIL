# HistoMIL
![HistoMIL](https://github.com/secrierlab/HistoMIL/blob/main/logo.png)

### Author: Shi Pan, UCL Genetics Institute

HistoMIL is a Python package for handling histopathology whole-slide images using multiple instance learning (MIL) techniques. With HistoMIL, you can create MIL datasets, train and evaluate MIL models, and make MIL predictions on new slide images.

## Getting Started

To use HistoMIL, you first need to create a conda environment with the required dependencies.

### create env with pre-defined file
You can do this by importing the env.yml file provided in this repository:

### linux user pre-requisites
1. Create conda env
```bash
conda create -n HistoMIL python=3.9
```
This will create a new environment named histomil, which you can activate with:

```bash
conda activate HistoMIL
```

### windows user pre-requisites

Windows (10+)
1. Download OpenSlide binaries from this page. Extract the folder and add bin and lib subdirectories to Windows system path. If you are using a conda environment you can also copy bin and lib subdirectories to [Anaconda Installation Path]/envs/YOUR ENV/Library/.

2. Install OpenJPEG. The easiest way is to install OpenJpeg is through conda using

```bash
conda create -n HistoMIL python=3.9
```
This will create a new environment named histomil, which you can activate with:

```bash
conda activate HistoMIL
```

```bash
C:\> conda install -c conda-forge openjpeg
```

### macOS user pre-requisites
On macOS there are two popular package managers, homebrew and macports.

Homebrew
```bash
brew install openjpeg openslide
```
MacPorts
```bash
port install openjpeg openslide
```

### create env manually 

Then install openslide and pytorch-gpu with following scripts.

```bash
conda install -c conda-forge openslide
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
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

HistoMIL is released under the GNU-GPL License. See the LICENSE file for more information.
