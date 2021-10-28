# simple utility script for turning models into meshes!
import os
import time

import numpy as np
import pandas as pd
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree

import binvox_rw as bv
import geometry as gm
import readTest
from NeuralImplicit import computeXtrain
from NeuralImplicit import NeuralImplicit as NeuralImplicitNoLatent
from NeuralImplicit_1 import NeuralImplicit
from extensions.SphereTracer import SphereTracer
from spherelatent import computePtLatentPredict



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def do_compute_chamfer(meshPath,weightPath,spherePath,outspherePath,fuspherenum,outpath,sphere32=None):
    trainedModels = list([f for f in os.listdir(weightPath)])
    cdlist = []

    for m in trainedModels:


        time0 = time.time()
        print('开始处理: {}'.format(m))
        modelPath = os.path.join(weightPath, m, '128grid_'+m + '.pth')
        meshFile = os.path.join(meshPath, m) + '.stl'
        sphereFile = os.path.join(spherePath, m) + '.txt'
        if outspherePath!=None:
            outsphereFile = os.path.join(outspherePath, m) + '.txt'
        else:
            outsphereFile = None
        ####################################################
        # sdfModel, spherelatent = loadModel(modelPath)
        # sdfModel.to(device)
        # sphereSampler = readTest.readSampler(meshFile, sphereFile, outsphereFile, 1000000)
        # sphere32 = sphereSampler.getSpheres()
        #################
        sdfModel, spherelatent,sphere32 = loadModelNoLatent(modelPath)
        sdfModel.to(device)
        sphereSampler = readTest.readSampler(meshFile, sphereFile, outsphereFile, 1000000)
        #######################################################
        sphtracer = SphereTracer(spherelatent,sphere32,fuspherenum)
        predPts = sphtracer.sample_surface(131072,sdfModel)
        predPts = predPts.cpu().numpy()
        mesh = sphereSampler.mesh
        surSampler = gm.PointSampler(mesh, ratio=0.0, std=0.0)
        gtPts = surSampler.sample(131072).astype(np.float32)
        gen_points_kd_tree = KDTree(predPts)
        one_distances, one_vertex_ids = gen_points_kd_tree.query(gtPts)
        gt_to_temp = np.square(one_distances)

        gt_to_gen_chamfer = np.mean(gt_to_temp)

        gt_points_kd_tree = KDTree(gtPts)
        two_distances, two_vertex_ids = gt_points_kd_tree.query(predPts)
        gen_to_gt_temp = np.square(two_distances)
        gen_to_gt_chamfer = np.mean(gen_to_gt_temp)
        print(np.max(gt_to_temp))
        print(np.max(gen_to_gt_temp))
        cd = gt_to_gen_chamfer + gen_to_gt_chamfer
        print(m,cd)
        cdlist.append([m,cd])
        print('elapsed time: {}'.format(time0-time.time()))
    data = pd.DataFrame(cdlist, columns=['name', 'cd'])
    data.to_csv(outpath, float_format='%.6f', index=False)




def compute_trimesh_chamfer(
    gtfile, genfile, num_mesh_samples=30000
):
    """
	This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

	gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see compute_metrics.ply
				for more documentation)

	gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction method
				(see compute_metrics.py for more)

	"""

    try:
        gt_mesh = trimesh.load(gtfile,force='mesh')
        gen_mesh = trimesh.load(genfile,force='mesh')

        gen_points_sampled = trimesh.sample.sample_surface(
            gen_mesh, num_mesh_samples
        )[0]
        gen_points_sampled = gen_points_sampled

        gt_points_np = gt_mesh.vertices

        gen_points_kd_tree = KDTree(gen_points_sampled)
        one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
        gt_to_temp = np.square(one_distances)
        gt_to_gen_chamfer = np.mean(gt_to_temp)

        gt_points_kd_tree = KDTree(gt_points_np)
        two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
        gen_to_gt_temp = np.square(two_distances)
        gen_to_gt_chamfer = np.mean(gen_to_gt_temp)
        return gt_to_gen_chamfer + gen_to_gt_chamfer
    except:
        print('error: '+os.path.split(genfile)[1])
        return -1



def computeIOU(gvoxels,tvoxels):
    t1 = gvoxels*tvoxels
    t2 = np.sum(t1)
    t3 = np.sum(gvoxels)+np.sum(tvoxels)-t2
    return t2/t3

def readBinvox(file):
    with open(file, 'rb') as f:
        model = bv.read_as_3d_array(f)

    return np.array(model.data,dtype=int)


def batchProcessIOU(meshPath,weightPath,spherePath,outspherePath,fuspherenum):
    trainedModels = list([f for f in os.listdir(weightPath)])
    ioulist = []
    for m in trainedModels:
        modelPath = os.path.join(weightPath, m,'8in_'+m + '.pth')
        meshFile = os.path.join(meshPath, m) + '.stl'
        sphereFile = os.path.join(spherePath, m) + '.txt'
        if outspherePath!=None:
            outsphereFile = os.path.join(outspherePath, m) + '.txt'
        else:
            outsphereFile = None
        sdfModel, spherelatent = loadModel(modelPath)
        sdfModel.to(device)
        spherelatent.to(device)
        sphereSampler = readTest.readSampler(meshFile, sphereFile, outsphereFile, 1000000)
        sdf = sphereSampler.sdf
        sphere32 = torch.from_numpy(sphereSampler.getSpheres()).cuda()
        mesh = sphereSampler.mesh
        sdfModel.eval()
        uniformSampler = gm.PointSampler(mesh, ratio=1.0, std=0.0)
        uniformPts = uniformSampler.sample(100000).astype(np.float32)
        #rspherePts = computePtLatentPredict(uniformPts, spherelatent, sphere32, fuspherenum, device)
        rspherePts = computeXtrain(torch.from_numpy(uniformPts).cuda(),sphere32[:,:3],spherelatent)
        with torch.no_grad():
            Pred = sdfModel(rspherePts).cpu().numpy()

        uniformGts = sdf.query(uniformPts)
        Pred = np.where(Pred<0,1,0)
        uniformGts = np.where(uniformGts<0,1,0)
        iou = computeIOU(Pred,uniformGts)
        ioulist.append([m,iou])

    return ioulist


# def batchProcessIOU(gvdir,tvdir):
#     res = []
#     for gv in os.listdir(gvdir):
#         if gv!= '1047582_sf.binvox':
#             continue
#         vname = os.path.splitext(gv)[0]
#         gfile = gvdir+gv
#         tfile = tvdir+gv
#         if not os.path.exists(gfile) or not os.path.exists(tfile):
#             print('error: ',gv)
#             continue
#
#         gvoxels = readBinvox(gfile)
#         tvoxels = readBinvox(tfile)
#         iou = computeIOU(gvoxels,tvoxels)
#         res.append([vname,iou])
#     return res

def writeIOUCsv(ioulist,outpath):
    df = pd.DataFrame(ioulist,columns=['name','iou'])
    df.to_csv(outpath,float_format='%.4f')


def loadModel(modelPath):

    sdfModel = NeuralImplicit.OverfitSDF(6, 32)
    weight_dict = torch.load(modelPath)
    sdfModel.load_state_dict(weight_dict['net'])
    spherelatent = weight_dict['spherelatent']
    return sdfModel,spherelatent

def loadModelNoLatent(modelPath):
    sdfModel = NeuralImplicitNoLatent.OverfitSDF(6, 32)
    sphereModel = NeuralImplicitNoLatent.SpherelatentNet()
    weight_dict = torch.load(modelPath)
    sphere32 = weight_dict['sphere32'].to(device)
    sdfModel.load_state_dict(weight_dict['net'])
  #  sphereModel.load_state_dict(weight_dict['sphereNet'])
    spherelatent = weight_dict['spherelatent']
    sphereModel.to(device)
    # with torch.no_grad():
    #     spherelatent = sphereModel(sphere32)
    return sdfModel, spherelatent,sphere32.cpu().numpy()

def writeErrorCsv(errorlist,outpath,flag):
    df = pd.DataFrame(errorlist,columns=['Name', 'Grid Error', 'Surface Error', 'Importance Error'])
    df.to_csv(outpath,float_format='%.4f',mode='a',header=flag)


def doevalIOU(meshPath,weightPath,spherePath,outspherePath,fuspherenum,outpath):
    ioulist = batchProcessIOU(meshPath,weightPath,spherePath,outspherePath,fuspherenum)
    writeIOUCsv(ioulist, outpath)

def doevalError(meshPath,weightPath,spherePath,outspherePath,outpath,fuspherenum,res=128):

    trainedModels = list([f for f in os.listdir(weightPath)])
    cubeMarcher = gm.CubeMarcher()
    uniformGrid = cubeMarcher.createGrid(res).astype(np.float32)
    errorlist = []

    index = 0
    for m in trainedModels:

        modelPath = os.path.join(weightPath, m, '128_'+m+'.pth')
        meshFile = os.path.join(meshPath, m) + '.stl'
        sphereFile = os.path.join(spherePath, m) + '.txt'

        outsphereFile = os.path.join(outspherePath, m) + '.txt' if outspherePath!=None else None
        sdfModel,spherelatent,sphere32 = loadModelNoLatent(modelPath)
        sdfModel.to(device)
        spherelatent.to(device)
        print("[INFO] Loaded model: ", m)
        print("[INFO] Loading mesh: ", m)
        #mesh = gm.Mesh(meshFile)

        print("[INFO] Inferring Grid")
        sphereSampler = readTest.readSampler(meshFile, sphereFile,outsphereFile, 1000000)
        sdf = sphereSampler.sdf
        # sphere32 = sphereSampler.getSpheres()
        sphere32 = torch.from_numpy(sphere32).cuda()
        mesh = sphereSampler.mesh

        sphereGrid = computeXtrain(torch.from_numpy(uniformGrid).cuda(),sphere32[:,:3],spherelatent)

        sdfModel.eval()
        with torch.no_grad():
            Pred = sdfModel(sphereGrid).cpu().numpy()
        # pointSampler = gm.ImportanceSampler(sphereSampler.mesh,
        #                                     int(epochlength / samplemethod['ratio']),
        #                                     samplemethod['weight'])
        # gridPred = sdfModel.predict(uniformGrid)

        Groundtrue = sdf.query(uniformGrid)
        gridError = np.mean(np.abs(Groundtrue - Pred))
        del(sphereGrid)
        print("[INFO] Inferring Surface Points")
        surfaceSampler = gm.PointSampler(mesh, ratio=0.0, std=0.0)
        surfacePts = surfaceSampler.sample(100000).astype(np.float32)
        rspherePts = computeXtrain(torch.from_numpy(surfacePts).cuda(),sphere32[:,:3],spherelatent)
        with torch.no_grad():
            Pred = sdfModel(rspherePts).cpu().numpy()

        surfaceError = np.mean(np.abs(Pred))
        del(surfaceSampler,surfacePts,rspherePts)
        print("[INFO] Inferring Importance Points")
        impSampler = gm.PointSampler(mesh, ratio=0.1, std=0.01)
        impPts = impSampler.sample(100000).astype(np.float32)
        rsphereImPts = computeXtrain(torch.from_numpy(impPts).cuda(),sphere32[:,:3],spherelatent)
        with torch.no_grad():
            Pred = sdfModel(rsphereImPts).cpu().numpy()



        Groundtrue = sdf.query(impPts)
        impError = np.mean(np.abs(Groundtrue - Pred))
        del(impSampler,impPts,rsphereImPts)


        print("[INFO] Grid Error: ", gridError)
        print("[INFO] Surface Error: ", surfaceError)
        print("[INFO] Imp Error (loss): ", impError)
        errorlist.append([m, gridError, surfaceError, impError])
        outdir = os.path.split(outpath)[0]
        outname = os.path.split(outpath)[1]
        # if (index + 1) % 5 == 0:
        #
        #     writeErrorCsv(errorlist, outpath,index==4)
        #
        #     errorlist = []
        #
        # index += 1

    writeErrorCsv(errorlist, outpath,True)

# def doevalChamferDistance(meshPath,weightPath,spherePath,outspherePath,fuspherenum):
#     #gtpath = '/home/zany2333/桌面/selectData_normal/'
#     cdlist = []
#     count = 0
#     for m in os.listdir(genpath):
#         genfile = os.path.join(genpath, m, m + '.obj')
#         gtfile = os.path.join(gtpath, m + '.obj')
#         cd = compute_chamfer()
#         cdlist.append([m, cd])
#         count += 1
#         # print(str(count)+' : ',m)
#     data = pd.DataFrame(cdlist, columns=['name', 'cd'])
#     data.to_csv(outpath, float_format='%.4f', index=False)
#


if __name__ == "__main__":
    # this should handle folders of meshes, parallelizing the meshing to avail cores
    # gvdir = './voxels/'
    # tvdir = '/home/cscv/桌面/compare/trueVoxels/'
    # iou_outpath = '/home/cscv/桌面/compare/64_4*64_2_3000e_32768_1e-2patience/iou.csv'
    # meshPath = '/home/cscv/桌面/selectData/'
    # weightPath = './results/64_4*64_2_3000e_32768_1e-2patience_2/'
    # spherePath = './sphere/selectDataSphere/'
    # outspherePath = './sphere/selectDataSphereOut/'
    # error_outpath = '/home/cscv/桌面/compare/64_4*64_2_3000e_32768_1e-2patience/error2.csv'
    # genpath = '/home/cscv/桌面/overfit_latent/results/64_4*64_10_cos/'
    # gtpath = '/home/cscv/桌面/selectData_normal/'
    # cd_outpath = '/home/cscv/桌面/compare/64_4*64_10_cos_compare/cd.csv'

    #doevalIOU(gvdir, tvdir, iou_outpath)
   # doevalChamferDistance(genpath, gtpath, cd_outpath)
   #  doevalError(meshPath,weightPath,spherePath,outspherePath,error_outpath)
    weiP = '/home/cscv/桌面/overfit_latent/results/thingi32Uniform/'
    meshP = '/home/cscv/桌面/thingi32/'
    spherePath = '/home/cscv/桌面/overfit_latent/sphere/thingi10kSphere/insphere128/'
    #outspherePath = '/home/cscv/桌面/overfit_latent/sphere/thingi10kSphere/outsphere128/'
    iououtpath = '/home/cscv/桌面/selectDataGTmesher/thingi32_iou_8in.csv'
    cdoutpath = '/home/cscv/桌面/selectDataGTmesher/thingi8_cd_gridnew.csv'
    erroroutpath = '/home/cscv/桌面/selectDataGTmesher/thingi32_sph_error.csv'
    # import threading
    # t1 = threading.Thread(target=do_compute_chamfer,args=(meshP,weiP,spherePath,outspherePath,32,cdoutpath))
    # t2 = threading.Thread(target=doevalIOU,args=(meshP,weiP,spherePath,outspherePath,32,iououtpath))
    # ts = [t1,t2]
    # for t in ts:
    #     t.setDaemon(True)
    #     t.start()
    # for t in ts:
    #     t.join()
    #do_compute_chamfer(meshP,weiP,spherePath,None,64,cdoutpath)
    #doevalIOU(meshP,weiP,spherePath,None,64,iououtpath)
    doevalError(meshP,weiP,spherePath,None,erroroutpath,32,32)



            # if index % 20 == 0:
            #     del sdfModel,mesh,surfaceSampler,surfacePts,surfacePred,impSampler,impPts,impPred,sdf,impTrue
            #     gc.collect()
            #     print("+++++++++++++++++++++++++ram clear+++++++++++++++++++++++++++++")








