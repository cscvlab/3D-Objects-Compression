# High-fidelity 3D Model Compression based on Key Spheres

![image](imgs/Figure1.png)

This repository contains the implementation of the paper:

High-fidelity 3D Model Compression based on Key Spheres
**DCC 2022 (Oral)**

Training a specific network for each 3D model to predict the signed distance function(SDF), which individually embeds its shape,  can realize compressed representationand reconstruction of objects by storing fewer network (and possibly latent) parame-ters.  However, it is difficult for the state-of-the-art methods NI [1] and NGLOD [2] toproperly reconstruct complex objects with fewer network parameters.  The method-ology we adopt is to utilize explicit key spheres [3] as network input to reduce thedifficulty of fitting global and local shapes.  By inputting the spatial information ofmultiple spheres which imply rough shapes (SDF) of an object, the proposed methodcan significantly improve the reconstruction accuracy with a negligible storage cost.An example is shown in Fig. 1.  Compared to the previous works, our method achievesthe high-fidelity and high-compression 3D object coding and reconstruction.

[1]  Thomas Davies,  Derek Nowrouzezahrai,  and Alec Jacobson,  “On the effectiveness ofweight-encoded neural implicit 3d shapes,” arXiv:2009.09808, 2020.
[2]  Towaki  Takikawa,  Joey  Litalien,  Kangxue  Yin,  Karsten  Kreis,  Charles  Loop,  DerekNowrouzezahrai, Alec Jacobson, Morgan McGuire, and Sanja Fidler, “Neural geometriclevel of detail:  real-time rendering with implicit 3d shapes,”  inCVPR, 2021.
[3]  Siyu Zhang, Hui Cao, Yuqi Liu, Shen Cai, Yanting Zhang, Yuanzhan Li, and XiaoyuChi,   “Sn-graph:  a  minimalist  3d  object  representation  for  classification,”   inICME,2021.

## Network
We propose that key spheres can be used as constraints to predict SDF values. The figure below shows the theory of our method and the difference from other methods in 2D image.
![image](imgs/figure2_git.png)
This is our network structure.
![image](imgs/network.png)
## Experiment
![image](imgs/figure6_git.png)
![image](imgs/table1_git.png)

##Results
For each mesh model, we generate the corresponding network model as its compression result. We also provide some reconstructed mesh models for display. The mesh reconstruction mesh models shown are all reconstructed using the 128-resolution marching cube algorithm, the training setup are 6*32 MLP, 128 key balls, and no hidden vectors are used. You can find them under the results folder.
![image](imgs/figure1_1.gif)
![image](imgs/figure1_2.gif)
![image](imgs/figure1_3.gif)


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

## Explanation
1. NeuralImplicit.py corresponds to the first architecture in the paper, NeuralImplicit_1.py corresponds to the second architecture.
2. We provide ball files for thingi10k objects.

## Third-Party Libraries

This code includes code derived from 3 third-party libraries

https://github.com/nv-tlabs/nglod
https://github.com/u2ni/ICML2021

## License
This project is licensed under the terms of the MIT license (see LICENSE for details).
