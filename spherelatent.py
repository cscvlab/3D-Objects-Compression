
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getDistance(a,b):
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)
    bt = b.t()
    return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))


def computePropotionCuda(pts,sphere32,fuspherenum):




    pts = torch.from_numpy(pts).to(device)
    sphere32 = torch.from_numpy(sphere32).to(device)
    sphere32_xyz = sphere32[:, :3]
    # temp_dis = sphere32_xyz - pt
    # d32 = np.sqrt(np.square(temp_dis[:, 0]) + np.square(temp_dis[:, 1]) + np.square(temp_dis[:, 2]))

    sphereR = sphere32[:, 3]
    # if np.max(d32 - sphereR) >= 0:
    #     fd32 = d32 - sphereR
    #     fdmin = np.min(fd32)
    #     if fdmin>0.5:
    #         proportion = np.zeros((1,32),dtype=np.float32)
    #     else:
    #         proportion = 1 - (fd32 / np.sum(fd32))
    #         proportion = proportion[np.newaxis,:]
    #         proportion = proportion.astype(np.float32)
    # else:
    #     proportion = np.zeros((1,32),dtype=np.float32)
    #     i = np.argmin(d32 - sphereR)
    #     proportion[0,i] = 1.0
    spheredis = torch.unsqueeze(pts-sphere32_xyz[0],dim=0)

    for i in range(1,sphere32_xyz.shape[0]):
        spt = sphere32_xyz[i]
        spheredis = torch.cat((spheredis,torch.unsqueeze(pts-spt,dim=0)),dim=0)

    # spheredis = np.sqrt(np.sum(np.square(np.array(spheredis)), axis=2)).T - sphereR
    spherefacedis = torch.sqrt(torch.sum(torch.square(spheredis), dim=2)).t() - sphereR
    spheredis = torch.sqrt(torch.sum(torch.square(spheredis), dim=2)).t()
    # spheredis[spheredis == 0] = -1 * (1e-6)
    spheredis[spheredis == 0] = 1 * (1e-6)
    spheredisT = 1 / spheredis

    for index, sdis in enumerate(spheredisT):
        # maxdisTin = np.max(np.fabs(spherefacedis[index][:int(sdis.shape[0]/2)]))
        maxdisTin = torch.max(torch.abs(spherefacedis[index]))
        mindisT = torch.min(spherefacedis[index])
        if maxdisTin < 5:
            sdis[range(sdis.shape[0])] = 0
            continue
        elif mindisT < 0:
            sdis[torch.argmin(sdis)] *= (2)
            # sdis[sdis>0] = 0
            # continue
        if fuspherenum < sdis.shape[0]:
            sdis[torch.argsort(sdis)[:-1 * fuspherenum]] = 0

    spheredis_sum = torch.sum(spheredis, dim=1)
    spheredis_sum[spheredis_sum == 0] = 1e-6
    proportion = spheredis / torch.unsqueeze(spheredis_sum,dim=1)
    #proportion = proportion.astype(np.float32)

    return proportion


def computePropotion(pts,sphere32,fuspherenum):


    sphere32_xyz = sphere32[:, :3]
    sphereR = sphere32[:, 3]
    spheredis = []
    for spt in sphere32_xyz:
        spheredis.append(pts - spt)

    #spheredis = np.sqrt(np.sum(np.square(np.array(spheredis)), axis=2)).T - sphereR
    #spherefacedis = np.sqrt(np.sum(np.square(np.array(spheredis)), axis=2)).T - sphereR
    spheredis = np.sqrt(np.sum(np.square(np.array(spheredis)), axis=2)).T
    spheredis1 = getDistance(torch.from_numpy(pts),torch.from_numpy(sphere32_xyz))

    spheredis[spheredis == 0] = 1 * (1e-6)
    #disave = np.average(spheredis)
    #spheredis = 1 / spheredis

    #spheredis = np.square(1/spheredis)



    # spheredisT = 1 / spheredis
    #
    #
    #
    # #mindis = np.min(spherefacedis[:,:int(spherefacedis.shape[1]/2)],axis=1)
    # #仅使用内球情况下
    # maxdis = np.min(spherefacedis, axis=1)
    #mindis = np.min(spheredis, axis=1)
    #spheredis[mindis>0.5] = 0
    #spheredisT[mindis<0][range(spheredisT[mindis<0].shape[0]),np.argmin(spherefacedis[mindis<0],axis=1)]*=2

    # spheredisIndex = np.argsort(spheredis,axis=1)
    # spheredisBool = np.ones(spheredis.shape)
    # for inx,b in enumerate(spheredisBool):
    #      b[spheredisIndex[inx][-fuspherenum:]] = 0
    # spheredis[spheredisBool.astype(bool)] = 1e-6

    # for index,sdis in enumerate(spheredisT1):
    #     #maxdisTin = np.max(np.fabs(spherefacedis[index][:int(sdis.shape[0]/2)]))
    #     #mindisin = np.min(np.fabs(spherefacedis[index]))
    #     mindisT = np.min(spherefacedis[index])
    #     if mindisT>0.2:
    #         sdis[range(sdis.shape[0])] = 0
    #         continue
    #     elif mindisT<0:
    #
    #         sdis[np.argmin(spherefacedis[index])]*=(2)
    #         # sdis[sdis>0] = 0
    #         # continue
    #     if fuspherenum<sdis.shape[0]:
    #         sdis[np.argsort(sdis)[:-1*fuspherenum]] = 0
    spheredis_sum = np.sum(spheredis,axis=1)
    spheredis_sum[spheredis_sum==0] = 1e-6
    proportion = spheredis / spheredis_sum[:,np.newaxis]
    proportion = proportion.astype(np.float32)
    return torch.from_numpy(proportion)


def computePtLatent(pt,spherelatent,sphere32):
    # spherelatentnp = [it.detach().numpy()for it in spherelatent]
    # spherelatentnp = np.array(spherelatentnp)

    ptnp = pt.numpy()

    sphere32_xyz = sphere32[:, :3]
    temp_dis = sphere32_xyz - ptnp
    d32 = np.sqrt(np.square(temp_dis[:, 0]) + np.square(temp_dis[:, 1]) + np.square(temp_dis[:, 2]))
    sphereR = sphere32[:,3]
    if np.max(d32-sphereR)>=0:
        fd32 = d32-sphereR
        proportion = 1-(fd32/np.sum(fd32))


        proportion = torch.from_numpy(proportion)
        # print(proportion)
        # print(proportion.shape)
        # print(spherelatent)
        t = spherelatent[0]*proportion[0]
        for i in range(1,proportion.shape[0]):
            t = torch.add(t,spherelatent[i]*proportion[i])

        return torch.cat((pt,t),dim=0)
    else:
        i = np.argmin(d32-sphereR)
        return torch.cat((pt,spherelatent[i]),dim=0)






def computePtLatentPredict(grid,spherelatent,sphere32,fuspherenum,device):


    pro = computePropotion(grid,sphere32,fuspherenum).to(device)
    #pro = computePropotionCuda(grid, sphere32, fuspherenum)
    latent = spherelatent.t().mm(pro.t()).t()
    res = torch.cat((torch.from_numpy(grid).to(device),latent),dim=1)
    return res

def computePtLatentBatch(ptbatch,propotion,spherelatent,batch_ids,batchsize,samplenum):
    istart = batch_ids * batchsize
    if((batch_ids+1)*batchsize>=samplenum):
        iend = samplenum
    else:
        iend = istart+batchsize
    propotionbatch = propotion[istart:iend]
    res = spherelatent.t().mm(propotionbatch.t()).t()
    res = torch.cat((ptbatch,res),dim=1)
    return res
