import argparse
import os

import numpy as np

import geometry as geo


def save_sdf(sdfs,surfacePts,path,type='txt'):
    if type == 'txt':
        with open(path,'ab') as f:
            np.savetxt(path, sdfs*64, delimiter=' ',fmt='%.4f')
            np.savetxt(f,(surfacePts+1)*64,delimiter=' ',fmt='%.4f')
    elif type == 'npy':
        np.save(sdfs,path)
    else:
        print("ni zai xiang xiang?")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='read mesh and compute its sdf')
    parser.add_argument('--meshpath', type=str,
                        help="path of your mesh",default='/home/cscv/桌面/compareData/')
    parser.add_argument('--res', type=int, help='resolution', default=128)
    parser.add_argument('--outpath',help='can generate txt file of sdfs in this path', type=str,default='/media/cscv/cscvlab128/sdfOutput/')
    parser.add_argument('--surfacenum', help='surface point number', type=int, default=2048)
    args = parser.parse_args()
    surfacenum = args.surfacenum
    count = 0
    for m in os.listdir(args.meshpath):
        # count+=1
        # if count<=2325:
        #     continue
        meshfile = args.meshpath+m
        mesh = geo.Mesh(meshPath=meshfile)
        sdf = geo.SDF(mesh)
        cuber = geo.CubeMarcher()
        grid = cuber.createGrid(args.res)
        gridsdfs = sdf.query(grid)
        gridsdfs = np.squeeze(gridsdfs,axis=1)
        surfaceSampler = geo.PointSampler(mesh, ratio=0.0, std=0.0)
        surfacePts = surfaceSampler.sample(surfacenum)
       # sdfs = np.concatenate((gridsdfs,surfacePts.ravel()),axis=0)
        path = args.outpath + os.path.splitext(m)[0] + '_sdf.txt'
        save_sdf(gridsdfs,surfacePts,path)
        print('succeed in saving {}'.format(os.path.splitext(m)[0]))
