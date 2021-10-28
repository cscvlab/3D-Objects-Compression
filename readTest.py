import math

import numpy as np

import geometry as geo


# print(sampleVoxels[0:10])
# print(np.true_divide(sampleVoxels,4)[0:10])
class readSampler():
    def __init__(self,meshfile,spherefile,outspherefile,samplenum,spherenum=32,res=128):
        self.meshfile = meshfile
        self.spherefile = spherefile
        self.outspherefile = outspherefile
        if not meshfile==None:
            self.mesh = geo.Mesh(self.meshfile)
            self.sdf = geo.SDF(self.mesh)
        self.spherenum =spherenum
        self.res = res
        self.samplenum=samplenum

    def computeDisConstant(self,a,sphere32,spherenum=32):
        sphere32_xyz = sphere32[:, :3]
        sphere32_r = sphere32[:, 3]
        sphere32_r = sphere32_r[:, np.newaxis]
        temp_dis = sphere32_xyz - a
        d = np.sqrt(np.square(temp_dis[:, 0]) + np.square(temp_dis[:, 1]) + np.square(temp_dis[:, 2]))
        d = d[:, np.newaxis]
        t_dis = np.concatenate((sphere32_r, d), axis=1)

        #t_dis = t_dis[t_dis[:, 1].argsort()]
        #t_dis = t_dis[:spherenum, :]
        #r1 = t_dis[:, 0]
        r2 = t_dis[:, 1]
        #r3 = r1 / r2
        # gra_dis =  (r3 - np.min(r3)) / (np.max(r3) - np.min(r3))
        return np.concatenate((a, r2))

    def computeDis(self,a, sphere32,spherenum):
        sphere32_xyz = sphere32[:,:3]
        temp_dis = sphere32_xyz - a
        d = np.sqrt(np.square(temp_dis[:,0])+np.square(temp_dis[:,1])+np.square(temp_dis[:,2]))
        d = d[:,np.newaxis]
        temp_dis = np.concatenate((temp_dis,d),axis=1)
        temp_dis = temp_dis[temp_dis[:, 3].argsort()]
        temp_dis = temp_dis[:spherenum,:3]
        return np.concatenate((a,np.reshape(temp_dis,(-1,))), axis=0)

    def getSphereR(self,num):
        sphereR = self.sphere32[-1*self.sphere32[:,3].argsort()]
        return sphereR[:num,3]

    def computeDisandRadius(self,a, sphere32,spherenum):
        sphere32_xyz = sphere32[:,:3]
        sphere32_r = sphere32[:,3]
        sphere32_r = sphere32_r[:,np.newaxis]
        temp_dis = sphere32_xyz - a
        d = np.sqrt(np.square(temp_dis[:,0])+np.square(temp_dis[:,1])+np.square(temp_dis[:,2]))
        d = d[:,np.newaxis]
        t_dis = np.concatenate((sphere32_r,d),axis=1)

        #t_dis = t_dis[t_dis[:, 1].argsort()]
        t_dis = t_dis[:spherenum,:]
        r1 = t_dis[:,0]
        r2 = t_dis[:,1]+1e-6
        r3 = r1/r2
       # gra_dis =  (r3 - np.min(r3)) / (np.max(r3) - np.min(r3))
        return np.concatenate((a,r3))

    #
    # def computeDis(a, sphere32):
    #     sphere32_xyz = sphere32[:,:3]
    #     temp_dis = sphere32_xyz - a
    #     return np.concatenate((a,np.reshape(temp_dis,(-1,))), axis=0)

    def normalSamples(self,samples):
        _range = np.max(abs(samples))
        res = np.true_divide(samples,_range)
        return res

    def compute32(self,sampleVoxels,sdf_samples):

        samples = []
        if sdf_samples.shape[0]!=1:
            for i in range(sampleVoxels.shape[0]):

                samples.append(np.append(self.computeDis(sampleVoxels[i],self.sphere32,self.spherenum),sdf_samples[i]))
        else:
            for i in range(sampleVoxels.shape[0]):
                samples.append(self.computeDis(sampleVoxels[i], self.sphere32, self.spherenum))
        return np.array(samples)


    def getSDF(self,item):
        try:
            sdf_value = float(item)
            if sdf_value >= 0:
                return math.sqrt(sdf_value)
            else:
                return -1 * math.sqrt(math.fabs(sdf_value))
        except ValueError:
            print(item)


    def getSurfaceSamples(self,ratio,std,surfacenum):
        surfaceSampler = geo.PointSampler(self.mesh,ratio,std)
        surfacePoints = surfaceSampler._surfaceSamples(surfacenum)
        surfaceSdfs = self.sdf.query(surfacePoints)
        surfaceSamples = self.compute32(surfacePoints,surfaceSdfs)

        return surfaceSamples



    # def getVoxels(file):
    #     with open(file, 'r') as f:
    #         sdfstr = f.read()
    #     sdfstr = sdfstr.split(" ")
    #     sdfs = [getSDF(item) for item in sdfstr]
    #     voxels = np.zeros((128*128*128,5))
    #     for x in range(128):
    #         for y in range(128):
    #             for k in range(128):
    #                 index = x*128*128+y*128+k
    #                 voxels[index,0] = x
    #                 voxels[index,1] = y
    #                 voxels[index,2] = k
    #                 if sdfs[index]>=0:
    #                     voxels[index,3] = 1
    #                     voxels[index, 4] = sdfs[index]
    #                 else:
    #                     voxels[index,3] = -1
    #                     voxels[index,4] = -1 * sdfs[index]
    #
    #
    #
    #     return voxels[np.lexsort(voxels.T)]

    def getVoxels(self,res=128):


        cuber = geo.CubeMarcher()
        grid = cuber.createGrid(res)
        gridsdfs = self.sdf.query(grid)
        voxels = np.concatenate((grid, gridsdfs), axis=1)
        signv = voxels[:, 3] / np.fabs(voxels[:, 3])
        signv = signv[:, np.newaxis]
        voxels[:, 3] = np.fabs(voxels[:, 3])
        voxels = np.concatenate((voxels, signv), axis=1)
        voxels = voxels[voxels[:, 3].argsort()]
        #voxels = voxels.astype(np.float32)
        return voxels

    def sampleNumber(self,voxels):
        mid = 1000000
        sampleNum = self.samplenum
        sampleVoxels = None
        sdf_samples = None
        sampleVoxels = voxels[mid - sampleNum:mid, :3]
        sdf_samples = voxels[mid - sampleNum:mid, 4] * voxels[mid - sampleNum:mid, 3]
        # for index in range(voxels.shape[0]):
        #     if voxels[index,4]!=0:
        #         sampleNum+=index
        #         sampleVoxels = voxels[mid-sampleNum:mid,:3]
        #         sdf_samples = voxels[mid-sampleNum:mid,4]*voxels[mid-sampleNum:mid,3]
        #         break
        # print(sdf_samples[np.argmax(sdf_samples)])
        # print(sdf_samples[np.argmin(sdf_samples)])
        # sampleVoxels = (sampleVoxels-64)/64
        # sdf_samples  = sdf_samples/64
        return sampleVoxels, sdf_samples

    def getSpheres(self):

        with open(self.spherefile, 'r') as f:
            tspheres = f.readlines()
        spherenum = int(tspheres[0])
        if not self.outspherefile==None:
            sphere32 = np.zeros([spherenum*2, 4], dtype=np.float32)
            with open(self.spherefile,'r') as f:
                tspheres = f.readlines()
                for i in range(1,spherenum+1):
                    t = tspheres[i].split(' ')
                    sphere32[i-1] = [float(item) for item in t][:4]
                # sphere32[:,:3] = (sphere32[:,:3]-64)/64
                # sphere32[:,3] = sphere32[:,3]/64
                self.sphere32 = sphere32
            with open(self.outspherefile,'r') as f:
                tspheres = f.readlines()
                for i in range(1,spherenum+1):
                    t = tspheres[i].split(' ')
                    sphere32[i+spherenum-1] = [float(item) for item in t][:4]
                sphere32[:,:3] = (sphere32[:,:3]-64)/64
                sphere32[:,3] = sphere32[:,3]/64
                self.sphere32 = sphere32
        else:
            sphere32 = np.zeros([spherenum , 4], dtype=np.float32)
            with open(self.spherefile,'r') as f:
                tspheres = f.readlines()
                for i in range(1,spherenum+1):
                    t = tspheres[i].split(' ')
                    sphere32[i-1] = [float(item) for item in t][:4]
                sphere32[:,:3] = (sphere32[:,:3]-64)/64
                sphere32[:,3] = sphere32[:,3]/64
                self.sphere32 = sphere32

        return sphere32


    def doSample(self):
        self.getSpheres()
        voxels = self.getVoxels()
        sampleVoxels, sdf_samples = self.sampleNumber(voxels)
        #sdf_samples_abs = np.fabs(sdf_samples)
        samples = self.compute32(sampleVoxels,sdf_samples)
        # surfaceSamples = self.getSurfaceSamples(ratio=0.1,std=0.01,surfacenum=2)
        # samples = np.concatenate((voxelSamples,surfaceSamples),axis=0)
        np.random.shuffle(samples)
        return samples

# voxelfile = './myinput/old/voxel/th_airplane.txt'
# voxels = getVoxels(voxelfile)
# sampleVoxels, sdf_samples = sampleNumber(voxels,1000000)
