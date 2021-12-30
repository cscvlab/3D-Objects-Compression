# High-fidelity 3D Model Compression based on Key Spheres

![image](imgs/Figure1.png)

Comparison of three implicit neural reconstruction methods (NI[1], NGLOD[2] and OURS), a traditional mesh simplification method (QECD[3]) and the ground truth (GT). The number of storage parameters is shown in the parentheses. An MLP network with 8 hidden layers and $32$ nodes in each layer is used in NI.	The 32-dimensional latent vectors of 125 grid points and 4737 network parameters are stored in NGLOD. In QECD, the number of vertices and facets is manually controlled at about 2400, which represents about 7200 parameters. Three ground truth models with about 22K, 49K, 12K vertices and facets respectively, are shown in the fifth column from left to right. The 128 extracted key spheres[4] with 512 parameters, which are used as the input of our network, are displayed in the last column. Except for our method with the fewest parameters (7026), all other methods show obvious errors.

[1] Thomas Davies, Derek Nowrouzezahrai, and Alec Jacobson, “On the effectiveness of weight-encoded neural implicit 3d shapes,” arXiv:2009.09808, 2020.
[2] Towaki Takikawa, Joey Litalien, Kangxue Yin, Karsten Kreis, Charles Loop, Derek Nowrouzezahrai, Alec Jacobson, Morgan McGuire, and Sanja Fidler, “Neural geometric level of detail: real-time rendering with implicit 3d shapes,” in CVPR, 2021.
[3] Marco Tarini, Nico Pietroni, Paolo Cignoni, Daniele Panozzo, and Enrico Puppo, "Practical quad mesh simplication," CGF, 29(2), 407-418, 2010.
[4] Siyu Zhang, Hui Cao, Yuqi Liu, Shen Cai, Yanting Zhang, Yuanzhan Li, and Xiaoyu Chi, “Sn-graph: a minimalist 3d object representation for classification,” in ICME, 2021.


## Getting started

### Ubuntu and CUDA version

We verified that it worked on ubuntu18.04 cuda10.2

### Python dependencies
The easiest way to get started is to create a virtual Python 3.6 environment via our environment.yml:
```
conda env create -f environment.yml
conda activate torch_over
cd ./submodules/miniball
python setup.py install

```
### Training
```
python train_series.py
```

### Evaluation
```
python eval.py
```
If you want to generate a reconstructed mesh through the MC algorithm
```
python modelmesher.py 
```

## Introduction
1. NeuralImplicit.py corresponds to the first architecture in the paper, NeuralImplicit_1.py corresponds to the second architecture.
2. We provide ball files for thingi10k objects.

## Third-Party Libraries

This code includes code derived from 3 third-party libraries

https://github.com/nv-tlabs/nglod
https://github.com/u2ni/ICML2021

