import logging
import os
import random
import sys
import time

import matplotlib
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import geometry as gm
import meshDataset as md
import readTest
from spherelatent import computePropotion, getDistance

matplotlib.use('Agg')

from matplotlib import pyplot as plt
from operator import itemgetter


def computeXtrain(x,sphere32,spherelatent):
    spheredis = getDistance(x, sphere32[:,:3])
    W = spheredis
    pro = W / torch.unsqueeze(torch.sum(W, dim=1), dim=1)
    lt = spherelatent.t().mm(pro.t()).t()
    x_train = torch.cat((x,lt),dim=1)
    return x_train

def my_loss(ypred,ytrue):
    #l1 = 1+torch.log(1+50*torch.abs(ytrue-ypred))
    l2 = torch.abs(ytrue-ypred)

    loss = torch.mean(l2)
    return loss



def computePropotionMatrix(pts,sphere32,fuspherenum):

    proportion = computePropotion(pts,sphere32,fuspherenum)

    return proportion

def buildGenMesh(sdfModel,output,spheresampler,spherelatent,res=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cubeMarcher = gm.CubeMarcher()
    rGrid = torch.from_numpy(cubeMarcher.createGrid(res).astype(np.float32)).cuda()
    rlen = rGrid.shape[0]
    rsphereGrid1 = computeXtrain(rGrid[:int(rlen/2)],torch.from_numpy(spheresampler.sphere32).cuda(),spherelatent)
    rsphereGrid2 = computeXtrain(rGrid[int(rlen/2):],torch.from_numpy(spheresampler.sphere32).cuda(),spherelatent)
    rsphereGrid = torch.cat((rsphereGrid1,rsphereGrid2),dim=0)
    sdfModel.eval()
    with torch.no_grad():
        S = sdfModel(rsphereGrid).cpu()
    cubeMarcher.march(rGrid.cpu().numpy(), S.numpy())
    marchedMesh = cubeMarcher.getMesh()
    marchedMesh.save(output)


def plotTrainResults(losslist,output, show=False, save=True):
    legend = ['Train']
    loss_history = losslist
    plt.plot(loss_history)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    msg = 'final loss=' + str(losslist[-1]) + '\n'

    plt.text(len(loss_history) * 0.8, loss_history[-1], msg)
    plt.legend(legend, loc='upper left')

    if save:
        plt.savefig(output)
    if show:
        plt.show()

    plt.close()

def createSampler(spherefile,outspherefile, meshfile, samplenum,samplemethod,spherenum=32,epochlength=1000000):
  sphereSampler = readTest.readSampler(meshfile, spherefile,outspherefile,int(samplenum*0.001), spherenum)
  pointSampler = gm.ImportanceSampler(sphereSampler.mesh,
                                      int(epochlength/samplemethod['ratio']),
                                      samplemethod['weight'])

  return sphereSampler,pointSampler

class NeuralImplicit:
    def __init__(self,H=6, N=32,epochs=250):
        self.N = N
        self.H = H
        self.model = self.OverfitSDF(H, N)
        self.epochs = epochs
        self.lr = 5e-3
        self.batch_size = 1024
        self.boundary_ratio = 0.99
        self.trained = False
        self.sphere32 = None
        self.propotion = torch.empty([0,])

    # Supported mesh file formats are .obj and .stl
    # Sampler selects oversample_ratio * num_sample points around the mesh, keeping only num_sample most
    # important points as determined by the importance metric
    def encode(self, spherenum,spherefile,outspherefile, meshfile,fuspherenum, samplenum=1000000,verbose=True):

        if (verbose and not logging.getLogger().hasHandlers()):
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
            logging.getLogger().setLevel(logging.INFO)

        #get data
        meshname = os.path.splitext(os.path.split(meshfile)[1])[0]
        samplemethod = {
            'weight': 60,
            'ratio': 0.1,
            'type': 'Importance'
        }
        sphereSampler, pointSampler = createSampler(spherefile,outspherefile, meshfile, samplenum,samplemethod,spherenum)
        self.sphere32 = sphereSampler.getSpheres()
        sphere32R = np.fabs(sphereSampler.sdf.query(self.sphere32[:,:3].copy()))
        self.sphere32 = np.concatenate((self.sphere32[:,:3],sphere32R),axis=1)
        self.sphere32 = self.sphere32.astype(np.float32)

        dataset = md.MeshDataset(sphereSampler,pointSampler, samplenum,verbose)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size,
                                shuffle=False,num_workers=8,pin_memory=True)

        self.spheremodel = self.SpherelatentNet(spherenum,self.sphere32)

        print("显卡是否可用", torch.cuda.is_available())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("所用设备为: ", device)

        self.model.to(device)
        self.spheremodel.to(device)



        loss_func = nn.L1Loss(reduction='mean')
        optimizer = optim.Adam(
            [
                {
                    "params": self.model.parameters(),
                    "lr": self.lr,
                },
                {"params": self.spheremodel.parameters(),
                 "lr": self.lr,
                }
            ]
        )

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                      factor=0.7,
                                                      verbose=True,
                                                      min_lr=1e-6,
                                                      threshold=1e-4,
                                                      threshold_mode='abs',
                                                      patience=10,cooldown=5)

        losslist = []
        print('start train')
        #show model structuer
        #summary(self.spheremodel, (0, spherenum))
        datalist = list(dataloader)
        #预训练
        pretrains = []
        print('预训练开始')
        for _ in range(5):
            prelr = self.lr
            premodel = self.OverfitSDF(self.H, self.N).to(device)
            prespheremodel = self.SpherelatentNet(spherenum,self.sphere32).to(device)
            ################
            # prespherelatent = torch.ones(spherenum,29).normal_(0, 0.1).to(device)
            # prespherelatent.requires_grad = True
            # preoptimizer = optim.Adam(
            #     [
            #         {
            #             "params": premodel.parameters(),
            #             "lr": prelr,
            #         },
            #         {"params": prespherelatent,
            #          "lr": prelr,
            #          }
            #     ]
            # )
            ##################
            preoptimizer = optim.Adam(
                [
                    {
                        "params": premodel.parameters(),
                        "lr": prelr,
                    },
                    {"params": prespheremodel.parameters(),
                     "lr": prelr,
                     }
                ]
            )

            prescheduler = lr_scheduler.ReduceLROnPlateau(preoptimizer, 'min',
                                                          factor=0.7,
                                                          verbose=True,
                                                          min_lr=1e-5,
                                                          threshold=1e-4,
                                                          threshold_mode='abs',
                                                          patience=10,cooldown=5)
            #prescheduler = lr_scheduler.CosineAnnealingWarmRestarts(preoptimizer, T_0=150, T_mult=2000, eta_min=0,verbose=True)
            for _ in range(20):
                epoch_loss = 0
                batch_idx = 0
                random.shuffle(datalist)
                for batch_idx, (x_train,y_train) in enumerate(datalist):
                    x_train,y_train = x_train.to(device),y_train.to(device)
                    preoptimizer.zero_grad()
                    spheresample,spherelatent = prespheremodel(x_train)
                    spherelatent = spherelatent.to(device)
                    spheresample = spheresample.to(device)
                    x_train = torch.cat((x_train, spheresample), dim=1)
                    sphere_train = torch.cat((torch.from_numpy(self.sphere32[:,:3]).cuda(),spherelatent),dim=1)
                    x_train = torch.cat((sphere_train,x_train),dim=0)
                    y_train = torch.cat((torch.from_numpy(-1*self.sphere32[:,3]).cuda(),y_train),dim=0)


                    #sphere_pred = premodel(sphere_train).squeeze(-1)
                    y_pred = premodel(x_train).squeeze(-1)

                    loss = my_loss(y_pred,y_train)
                    epoch_loss += loss.item()
                    loss.backward()
                    preoptimizer.step()
                prescheduler.step(epoch_loss)
            pretrain = {
                'sdfModel': premodel.state_dict(),
                'sphereModel': prespheremodel.state_dict(),
                #'spherelatent': spherelatent,
                'scheduler': prescheduler.state_dict(),
                'optimizer':preoptimizer.state_dict(),
                'lr': prelr,
                'loss': epoch_loss/(batch_idx+1)
            }
            print(pretrain['loss'])
            pretrains.append(pretrain)

        pretrains.sort(key=itemgetter('loss'))
        prelosses = [item['loss'] for item in pretrains]
        self.model.load_state_dict(pretrains[0]['sdfModel'])
        self.spheremodel.load_state_dict(pretrains[0]['sphereModel'])
        #spherelatent = pretrains[0]['spherelatent']
        self.lr = pretrains[0]['lr']
        scheduler.load_state_dict(pretrains[0]['scheduler'])
        optimizer.load_state_dict(pretrains[0]['optimizer'])
        e = 20
        print('预训练loss为:{:.6f} , {:.6f} , {:.6f}'.format(prelosses[0],prelosses[1],prelosses[2]))
        print('正式训练开始')
        while e < self.epochs:
            batch_idx = 0
            time0 = time.time()
            epoch_loss = 0
            epoch_error = 0
            self.model.train(True)
            random.shuffle(datalist)
            for batch_idx, (x_train,y_train) in enumerate(datalist):

                x_train,y_train = x_train.to(device),y_train.to(device)
                optimizer.zero_grad()
                spheresample,spherelatent = self.spheremodel(x_train)
                spherelatent = spherelatent.to(device)
                x_train = torch.cat((x_train,spheresample),dim=1)
                sphere_train = torch.cat((torch.from_numpy(self.sphere32[:, :3]).cuda(), spherelatent), dim=1)

                x_train = torch.cat((sphere_train, x_train), dim=0)
                y_train = torch.cat((torch.from_numpy(-1 * self.sphere32[:, 3]).cuda(), y_train), dim=0)


                #sphere_pred = self.model(sphere_train).squeeze(-1)
                y_pred = self.model(x_train).squeeze(-1)

                #loss = loss_func(y_pred, y_train)
                error = loss_func(y_pred[spherenum:], y_train[spherenum:])
                loss = my_loss(y_pred,y_train)
                loss.backward()
                optimizer.step()



                epoch_error+=error.item()
                epoch_loss += loss.item()
                    # if e == self.epochs - 1 or e == 0:
                    #     with open('./spheretrainlog.txt', 'a') as f:
                    #         f.write(msg)
                    #         f.write('\n')

            # if (early_stop and epoch_loss < early_stop):
            #     break
            time1 = time.time()
            epoch_loss = epoch_loss / (batch_idx + 1)
            epoch_error = epoch_error / (batch_idx+1)
            scheduler.step(epoch_loss)
            if e+1==self.epochs or (e+1)%10==0:
                msg = '{}\tEpoch: {}:\t[{}/{}]\tepoch_loss: {:.6f}\tepoch_error: {:.6f}\telapse: {:.2f}'.format(
                    meshname,
                    e + 1,
                    len(dataset),
                    len(dataset),
                    epoch_loss,
                    epoch_error,
                    (time1-time0))
                logging.info(msg)

            losslist.append(epoch_error)
            e+=1

        #create a results folder for this mesh

        if not os.path.exists('./results/' + meshname):
            os.mkdir('./results/' + meshname)

        outputdir = './results/'+ meshname
        model_file = os.path.join(outputdir,str(spherenum)+'in_'+meshname+'.pth')
        image_file = os.path.join(outputdir,str(spherenum)+'in_'+meshname+'.png')
        genmesh_file = os.path.join(outputdir,str(spherenum)+'in_'+meshname+'.obj')
        spherelatent = self.spheremodel.state_dict()['spherelatent']
        self.trained = True
        state = {
            'net': self.model.state_dict(),
            'spherelatent': spherelatent
        }
        torch.save(state, model_file)
        plotTrainResults(losslist,image_file)
       # buildGenMesh(self.model,genmesh_file,sphereSampler,spherelatent)

    def load(self, state_file):

        self.model.load_state_dict(torch.load(state_file),strict=False)
        #self.trained = True
        print('succeed in loading model')
        return True


    # Returns weights in row major form
    def weights(self):
        self.model.to(torch.device("cpu"))
        weights = np.empty((0,))
        for weight_mat in list(self.model.state_dict().values())[::2]:
            weights = np.concatenate((weights, np.squeeze(weight_mat.numpy().reshape(-1, 1))))
        return weights

    # Returns biases in row major form
    def biases(self):
        self.model.to(torch.device("cpu"))
        biases = np.empty((0,))
        for bias_mat in list(self.model.state_dict().values())[1::2]:
            biases = np.concatenate((biases, bias_mat.numpy()))
        return biases

    def renderable(self):
        assert (self.trained)
        return (self.H, self.N, self.weights(), self.biases())

    # The actual network here is just a simple MLP
    class OverfitSDF(nn.Module):
        def __init__(self, H, N):
            super().__init__()
            assert (N > 0)
            assert (H > 0)

            # Original paper uses ReLU but I found this lead to dying ReLU issues
            # with negative coordinates. Perhaps was not an issue with original paper's
            # dataset?
            net = [nn.Linear(32, N), nn.LeakyReLU(0.1)]

            for _ in range(H - 1):
                net += [nn.Linear(N, N), nn.LeakyReLU(0.1)]
            #net += [nn.Linear(N*2, N), nn.LeakyReLU(0.1)]
            net += [nn.Linear(N, 1)]
            self.model = nn.Sequential(*net)

        def forward(self, x):
            x = self.model(x)
            output = torch.tanh(x)
            return output

    class SpherelatentNet(nn.Module):

        def __init__(self,spherenum,sphere32, H=1, N=29):
            super().__init__()
            assert (N > 0)
            assert (H > 0)

            # Original paper uses ReLU but I found this lead to dying ReLU issues
            # with negative coordinates. Perhaps was not an issue with original paper's
            # dataset?
            self.spherenum = spherenum
            self.spherelatent = nn.Parameter(torch.randn(spherenum,29) * 0.01)
            #self.latent = nn.Parameter(torch.randn(1, 29, 32 + 1, 32+ 1, 32 + 1) * 0.01)
            self.sphere32 = torch.from_numpy(sphere32).cuda()
            #net = [nn.Linear(99,64),nn.Sigmoid(),nn.Linear(64,32),nn.Sigmoid()]

            #self.model = nn.Sequential(*net)
            #self.pro = spheredis/torch.sum(spheredis,dim=1)
           #nn.init.normal_(self.model[0].weight,mean=0,std=0.1)

        def forward(self, x):

           # N = x.shape[0]

            sphere32_xyz = self.sphere32[:,:3]
           # sample_coords = x.reshape(1, N, 1, 1, 3)  # [N, 1, 1, 3]
            #sphere_coords = self.sphere32[:,:3].reshape(1,self.spherenum,1,1,3)
          #  sample = torch.nn.functional.grid_sample(self.latent, sample_coords,
                                 #  align_corners=True, padding_mode='border')[0, :, :, 0, 0].transpose(0, 1)
            # spherelatent = torch.nn.functional.grid_sample(self.latent, sphere_coords,
            #                        align_corners=True, padding_mode='border')[0, :, :, 0, 0].transpose(0, 1)


            # sphere32_xyz = sphere32_xyz.unsqueeze(dim=0).expand(x.shape[0],self.spherenum,3)
            # spheredis = (x-sphere32_xyz.transpose(0,1)).transpose(0,1).reshape(x.shape[0],-1)

            spheredis = getDistance(x,sphere32_xyz)

            W = spheredis
           # spheredis = getDistance(x,self.sphere32[:,:3])
            #W = spheredis/torch.exp(R)
            #W = spheredis
            pro = W/torch.unsqueeze(torch.sum(W,dim=1),dim=1)
            spheresample = self.spherelatent.t().mm(pro.t()).t()
            #output = torch.cat((sample,spheresample),dim=1)
            return spheresample,self.spherelatent




