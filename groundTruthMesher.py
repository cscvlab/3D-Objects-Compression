import os

import geometry as gm


def marchGroundTruthMesh(meshname,meshfile,res,outputDir):
    mesh = gm.Mesh(meshfile)

    cubeMarcher = gm.CubeMarcher()
    inferGrid = cubeMarcher.createGrid(res)
    sdfs = gm.SDF(mesh)
    S = sdfs.query(inferGrid)
    cubeMarcher.march(inferGrid, S)
    marchedMesh = cubeMarcher.getMesh()
    marchedMesh.save(os.path.join(outputDir,meshname + '.obj'))

if __name__ == '__main__':
    meshdir = '/home/cscv/桌面/shapenet150/'
    outputDir = '/home/cscv/桌面/selectDataGTmesher2/'
    for file in os.listdir(meshdir):
        meshname = os.path.splitext(file)[0]
        print(meshname)
        marchGroundTruthMesh(meshname,os.path.join(meshdir,file),128,outputDir)
   # binvoxToMesh()