import os
import shutil
import subprocess

import numpy as np
from sklearn.neighbors import KDTree

import geometry as geo

inputdir = '../results/5*64/'
mesh_dir = './mesh/'
vox_dir = './voxels/'
outputdir = './sphereply/'
#可视化32个球文件


def sampleSurfaceSphere(mesh):
    pointsampler = geo.PointSampler(mesh)
    # unisampler = geo.PointSampler(mesh,ratio=1.0)
    # unipts = unisampler.sample(10000000)
    # sdf = geo.SDF(mesh)
    # s = sdf.query(unipts)
    # unipts = np.concatenate((unipts,s),axis=1)
    # punipts = unipts[unipts[:,3]>=0]
    # nunipts = unipts[unipts[:,3]<0]




    surPts = pointsampler.sample(100000)
    tree = KDTree(surPts,leaf_size=60)
    bsidep = tree.query_radius(surPts,r=0.05)
    bsidepn = np.array([b.shape[0] for b in bsidep])
    rweight = bsidepn / (np.sum(bsidepn)/bsidepn.shape[0])
    spherePts = []
    passPts = set()
    for pi in range(surPts.shape[0]):
        if pi in passPts:
            continue
        passes = tree.query_radius(surPts[pi:pi+1],r=0.05*rweight[pi])
        passPts.update(passes[0])
        spherePts.append(surPts[pi])
    spherePts = np.array(spherePts)
    tree = KDTree(spherePts)
    D = tree.query(spherePts,2)[0][:,1]
    spherePts = np.concatenate((spherePts,D[:,np.newaxis]),axis=1)
    with open('61765_sf.txt','w')  as f:
        f.write('{}\n'.format(spherePts.shape[0]))
        for pt in spherePts:
            f.write('{} {} {} {:.6f} 0\n'.format(int(np.rint((pt[0]+1)*64)),int(np.rint((pt[1]+1)*64)),int(np.rint((pt[2]+1)*64)),pt[3]*64))
        f.write('0')
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
   #  flag = 0
   #  for m in os.listdir(mesh_dir):
   #      #if os.path.exists(mesh_dir+m+'.obj'):
   #      if m== '39165_sf.obj':
   #          flag=1
   #      if flag==0:
   #          continue
   #      if os.path.splitext(m)[1]=='.obj':
   #          mesh2binvox(m)
   meshfile = '/home/cscv/桌面/selectData/61765_sf.stl'
   mesh = geo.Mesh(meshfile)
   sampleSurfaceSphere(mesh)



