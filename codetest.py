import numpy as np
import torch

# model = NeuralImplicit.OverfitSDF(4,64)
# path = '/home/zany2333/桌面/overfit_latent/results/1772309_4sf/1772309_sf.pth'
# result = torch.load(path)
# model = model.load_state_dict(result['net'])
# spherelatent = result['spherelatent']
# print(model)
a = torch.Tensor([[1,2,3]])
b = np.ones((3,3))
print(b>0)
b[b>0] = 0
print(b)
# b = torch.Tensor([[3,2,1]])
# print(a.shape)
# vec = torch.ones(32,32).normal_(0, 0.1).to('cpu')
# vec.requires_grad = True
#
# print(vec)

# sphere32_xyz = np.random.randint(1,10,(5,2))
# # temp_dis = sphere32_xyz - pt
# # d32 = np.sqrt(np.square(temp_dis[:, 0]) + np.square(temp_dis[:, 1]) + np.square(temp_dis[:, 2]))
# #
# sphereR = np.random.rand(32)
# print(sphere32_xyz)
# print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
# tindex = np.arange(sphere32_xyz.shape[0])
# np.random.shuffle(tindex)
# sphere32_xyz = sphere32_xyz[tindex]
# print(sphere32_xyz)
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
# spheredis = []
# pts = np.random.rand(100,3)
# for spt in sphere32_xyz:
#     t = pts-spt
#     spheredis.append(pts - spt)
#
# spheredis = np.sqrt(np.sum(np.square(np.array(spheredis)), axis=2)).T - sphereR
# spheredis[spheredis == 0] = -1 * (1e-6)
# spheredis = 1 / spheredis
# for sdis in spheredis:
#     if np.min(sdis) < 0:
#         sdis[sdis > 0] = 0
#     else:
#         sdis[np.argsort(sdis)[:-2]] = 0
#
#
# proportion = spheredis / np.sum(spheredis, axis=1)[:,np.newaxis]
