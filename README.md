# 3D-protein-classification-of-RISurConv


## Installation
The environment required to run this project can be set up using the provided environment.yml file.  

Install the pointnet++ cuda operation library by running the following command:
```
cd pointops
python3 setup.py install
cd ..
```

### Dataset Description
The SHREC2025 dataset is based on the class assignment of protein surfaces, derived from MMSeqs2 (Steinegger et al., Nature Biotechnology 2017) clusters using the RCSB API. The dataset includes only experimental models (from X-ray and NMR) with more than 50 residues. The solvent-excluded surfaces (SES) of these proteins, along with physicochemical properties such as electrostatic potential, were calculated.

This dataset consists of 11,565 protein surfaces across 97 classes, which will be divided into a training set (80%) and a test set (20%). Specifically:

The training set contains 9,244 protein surfaces with corresponding ground truth class annotations.

The test set includes 2,321 protein surfaces, which are unannotated.

Class Distribution:
The class distribution of protein surfaces in the training set and test set for the 97 classes is illustrated below.

The electrostatic potential was calculated using the Treecode-Accelerated Boundary Integral Poisson-Boltzmann solver, implemented in APBS (Geng, Krasny, 2013). The surface models were generated using NanoShaper.

To download the datasets, please visit the official SHREC2025 website:https://shrec2025.drugdesign.fr/

### Training
```
python train_classification.py
```
The output results will be stored in the res/output.csv file.

Original link:https://github.com/cszyzhang/RISurConv

