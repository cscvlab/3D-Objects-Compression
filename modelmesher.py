# simple utility script for turning models into meshes!

import numpy as np
import torch
import os
import geometry as gm
import readTest

from eval import loadModel,loadModelNoLatent
from NeuralImplicit import computeXtrain

def buildGenMesh(sdfModel,output,sphereSampler,spherelatent,res=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cubeMarcher = gm.CubeMarcher()
    rGrid = torch.from_numpy(cubeMarcher.createGrid(res).astype(np.float32)).cuda()
    rlen = int(rGrid.shape[0]/8)
    rsphereGrid = computeXtrain(rGrid[:rlen], torch.from_numpy(sphereSampler.sphere32[:, :3]).to(device) , spherelatent)
    sdfModel.eval()
    with torch.no_grad():
        S = sdfModel(rsphereGrid).cpu()
        for i in range(1,8):
            print(i)
            S = torch.cat((S,sdfModel(computeXtrain(rGrid[rlen*i:rlen*(i+1)], torch.from_numpy(sphereSampler.sphere32[:, :3]).to(device) , spherelatent)).cpu()),dim=0)

    # rsphereGrid = computePtLatentPredict(rGrid,spherelatent,spheresampler.sphere32,fuspherenum,device)
    # S = S.numpy()
    cubeMarcher.march(rGrid.cpu().numpy(), S.numpy())
    marchedMesh = cubeMarcher.getMesh()
    marchedMesh.save(output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    # this should handle folders of meshes, parallelizing the meshing to avail cores

    # support both single neural implicit, and a folder
    wDir = '/home/cscv/桌面/overfit_latent/results/thingi32Uniform/'
    oDir = '/home/cscv/桌面/buildMesh/thingi32_128_nolatent/'
    mDir = '/home/cscv/桌面/thingi32/'
    sphDir = '/home/cscv/桌面/overfit_latent/sphere/thingi10kSphere/insphere128/'
    sphNum = 128
    # wFile = '/home/cscv/桌面/overfit_latent/results/472139/128_6_32_472139.pth'
    # mFile = '/home/cscv/桌面/hardData/thingi/472139.stl'
    # sphFile = '/home/cscv/桌面/overfit_latent/sphere/hardDataSphere/insphere128/472139.txt'
    # output = '/home/cscv/桌面/buildMesh/472139.obj'
    # sdfModel, spherelatent, sphere32 = loadModelNoLatent(wFile)
    # sdfModel.to(device)
    # spherelatent.to(device)
    # sphereSampler = readTest.readSampler(mFile, sphFile, None, 1000, sphNum)
    # sphereSampler.getSpheres()
    # buildGenMesh(sdfModel, output, sphereSampler, spherelatent)
    for w in os.listdir(wDir):
        wFile = os.path.join(wDir,w,'128_'+w+'.pth')
        mFile = os.path.join(mDir,w+'.stl')


        sphFile = os.path.join(sphDir,w+'.txt')
        output = os.path.join(oDir, w + '.obj')
        #sdfModel,spherelatent = loadModel(wFile)
        sdfModel, spherelatent,sphere32 = loadModelNoLatent(wFile)
        sdfModel.to(device)
        spherelatent.to(device)
        sphereSampler = readTest.readSampler(mFile, sphFile,None,1000, sphNum)
        sphereSampler.getSpheres()
        sphereSampler.sphere32 = sphere32
        buildGenMesh(sdfModel,output,sphereSampler,spherelatent)
    


