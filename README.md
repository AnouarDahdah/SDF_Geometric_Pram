# SDF-Based Geometry Parameterization and Neural Network Architectures

This repository is focused on extracting and processing data from CFD simulations in CGNS format, calculating its Signed Distance Function (SDF), and experimenting with various deep learning architectures to develop a neural network-parameterized geometry for further processing.

## Project Overview

The project explores using deep learning to parameterize geometries represented by SDFs. Our primary goals include geometry refinement, analysis, ROM analysis

1. **Data Extraction**:
   - We use `h5py` to extract XYZ geometry data from CGNS files, providing the structured CFD simulation data necessary for SDF calculation.

2. **Signed Distance Function Calculation**:
   - After extracting the geometry, we compute the Signed Distance Function (SDF) for each point, creating a continuous representation of distances .

3. **Neural Network Architectures**:
   - Various architectures are tested to develop a parameterized representation of the geometry, including:
     - **Autoencoder**: Compresses and reconstructs geometry data by learning latent representations of SDF values.
     - **Neural ODE (Ordinary Differential Equation)**: Captures continuous transformations within the geometry, refining latent representations learned by the autoencoder.

4. **Parameterization and Refinement**:
   - Our objective is to find effective parameterization techniques for representing the geometry in a compressed yet expressive format.



### Requirements

To run the code in this repository, you need:
- Python >= 3.x
- [PyTorch](https://pytorch.org/) for neural network architectures
- [h5py](http://www.h5py.org/) for CGNS data handling


