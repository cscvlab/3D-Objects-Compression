# Dataset generates data from the mesh file using the SDFSampler library on CPU
# Moving data generation to GPU should speed up this process significantly
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MeshDataset(Dataset):
    def __init__(self,sphereSampler,pointSampler,samplenum=1000000,verbose=True):


        meshname = os.path.split(sphereSampler.meshfile)[1]
        if (verbose):
            logging.info("Loaded " + meshname)

        # vertices, faces = MeshLoader.read(voxelFile)
        # normalizeMeshToSphere(vertices, faces)

        # sampler = PointSampler(vertices, faces)
        ##########################################################

        ##########################################################
        # Testing indicated very poor SDF accuracy outside the mesh boundary which complicated
        # raymarching operations.
        # Adding samples through the unit sphere improves accuracy farther from the boundary,
        # but still within the unit sphere
        #  general_points = sampler.sample(int((1-boundary_ratio)*num_samples), 1)
        #ptsNp = readTest.doSample(sphereFile, voxelFile, num_samples)
        queryPts = pointSampler.sample(int(samplenum * 1))
        S = sphereSampler.sdf.query(queryPts)
        trainData = np.concatenate((queryPts, S), axis=1)
        tindex = np.arange(trainData.shape[0])
        np.random.shuffle(tindex)
        self.trainData = torch.from_numpy(trainData).type(torch.float32)

    def getAllData(self):
        return self.trainData.numpy()


    def __getitem__(self, index):
        #return self.trainData[0,index, :3], self.trainData[:,index, 3]
        return self.trainData[index,:3],self.trainData[index,3]

    # def __getitem__(self, index):
    #     return torch.tensor([self.pts[index, :99]], dtype=torch.float32), torch.tensor([self.pts[index, 99]],
    #                                                                                    dtype=torch.float32)

    def __len__(self):
        return self.trainData.shape[0]