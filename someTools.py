import os
import shutil
import subprocess

import edt
import numpy as np

import binvox_rw
import geometry as gm

inputdir = './results/64_4*64_2_3000e_32768_1e-2patience/'
mesh_dir = './mesh/'
vox_dir = './voxels/'
outputdir = './sphereply/'


#分离已训练和未训练模型
def divideTrainedMesh():
    resultDir = '/home/cscv/桌面/overfit_latent/results/thingi10k6_32/'
    meshDir = '/media/cscv/cscvlab128/Thingi10K/used_meshes/'
    outputdir = '/media/cscv/cscvlab128/Thingi10K/trained_meshes/'
    for m in os.listdir(resultDir):
        meshfile = os.path.join(meshDir,m+'.stl')
        outputpath = os.path.join(outputdir,m+'.stl')
        shutil.move(meshfile,outputpath)

#将体素文件转为网格
def binvoxToMesh(voxfile,ooutputfile):
    with open(voxfile,'rb') as f:
        vox = binvox_rw.read_as_3d_array(f).data
    S = edt.edt(
        vox,
        black_border=True, order='F',
        parallel=4  # number of threads, <= 0 sets to num cpu
    )
    S = S.reshape(128**3,-1)
    cubeMarcher = gm.CubeMarcher()
    inferGrid = np.zeros((128**3,3))
    cont = 0
    for i in range(128):
        for j in range(128):
            for k in range(128):
                inferGrid[i*128*128+j*128+k] = np.array([i,j,k])
    cubeMarcher.march(inferGrid,S)
    marchedMesh = cubeMarcher.getMesh()
    marchedMesh.save(ooutputfile)

#可视化32个球文件
def generateSpherePly(filename):
    p = subprocess.Popen(["toplysdf.exe"], stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    p.stdin.write(inputdir+filename+'.txt\n')

    p.stdin.write(outputdir+filename+'.ply\n')

    p.stdin.write('32\n')
    p.communicate()
   # p.wait()
    print("done: "+filename)

#网格模型转体素
def mesh2binvox(filename):
    print("start: "+filename)
    voxname = os.path.splitext(filename)[0]+'.binvox'
    p = subprocess.Popen(["./binvox","-d","128","-cb",mesh_dir+filename],
                         stdout=subprocess.PIPE,
                         universal_newlines=True)
    msg = p.communicate()[0]
    if 'Error' in msg:
        shutil.move(mesh_dir+filename,'./error/'+filename)
        p.kill()
    else:
        os.remove(mesh_dir+filename)
        shutil.move(mesh_dir + voxname, vox_dir + voxname)
        print("done: " + filename)

def allToOne():
    for dirname in os.listdir(inputdir):
        for file in os.listdir(os.path.join(inputdir,dirname)):
            if os.path.splitext(file)[1] == '.obj':
                shutil.copy(os.path.join(inputdir,dirname,file),mesh_dir)
if __name__ == '__main__':

   # allToOne()
   #  vox_dir = '/home/cscv/桌面/tempdata/'
   #  outputdir = '/home/cscv/桌面/selectDataGTmesher/'
   #  for m in os.listdir(vox_dir):
   #      #if os.path.exists(mesh_dir+m+'.obj'):
   #      voxfile = os.path.join(vox_dir,m)
   #      outputfile = os.path.join(outputdir,m.split('.')[0]+'.obj')
   #      binvoxToMesh(voxfile,outputfile)
    divideTrainedMesh()


