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
import geometry as gm


def computeXtrain(x,sphere32_xyz,spherelatent):
    spheredis = getDistance(x, sphere32_xyz)
    #spheredis = 1/spheredis
    #spheredis[spheredis>1] = 1e-6
    pro = spheredis / torch.unsqueeze(torch.sum(spheredis, dim=1), dim=1)
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

def buildGenMesh(sdfModel,output,sphere32,spherelatent,res=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cubeMarcher = gm.CubeMarcher()
    rGrid = torch.from_numpy(cubeMarcher.createGrid(res).astype(np.float32)).cuda()
    rsphereGrid = computeXtrain(rGrid, sphere32[:, :3], spherelatent)
    sdfModel.eval()
    with torch.no_grad():
        S = sdfModel(rsphereGrid).cpu()
    # rsphereGrid = computePtLatentPredict(rGrid,spherelatent,spheresampler.sphere32,fuspherenum,device)
    # S = S.numpy()
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
  # pointSampler = gm.PointSampler(sphereSampler.mesh,samplemethod['ratio'],samplemethod['std'])
  importaceSampler = gm.ImportanceSampler(sphereSampler.mesh,
                                      int(epochlength/samplemethod['ratio']),
                                      samplemethod['weight'])


  return sphereSampler,importaceSampler

class NeuralImplicit:
    def __init__(self,H=6, N=32,epochs=150):
        self.N = N
        self.H = H
        self.model = self.OverfitSDF(H, N)
        self.spheremodel = self.SpherelatentNet()
        self.epochs = epochs
        self.lr = 5e-3
        self.batch_size = 512
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
        #########################################################################################

        #sphere or points
        # sphere
        self.sphere32 = sphereSampler.getSpheres()
        sphere32R = sphereSampler.sdf.query(self.sphere32[:, :3].copy())
        self.sphere32 = np.concatenate((self.sphere32[:, :3], sphere32R), axis=1)
        self.sphere32 = self.sphere32.astype(np.float32)


        #######################################################################################################
        # surface points
        # surface points(ration=0,std=0.005) random points(ratio=1.0,std=0)
        # pointSampler = gm.PointSampler(sphereSampler.mesh, ratio=1.0, std=0)
        # self.sphere32 = pointSampler.sample(spherenum)
        # sphere32R = sphereSampler.sdf.query(self.sphere32[:, :3].copy())
        #
        # self.sphere32 = np.concatenate((self.sphere32, sphere32R), axis=1)
        # self.sphere32 = self.sphere32.astype(np.float32)

        #uniform points
        # cuber = gm.CubeMarcher()
        # self.sphere32 = cuber.createGrid(5)
        # sphere32R = sphereSampler.sdf.query(self.sphere32[:,:3].copy())
        # self.sphere32 = np.concatenate((self.sphere32, sphere32R), axis=1)
        # self.sphere32 = self.sphere32.astype(np.float32)
        #########################################################################################


        dataset = md.MeshDataset(sphereSampler,pointSampler, samplenum,verbose)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size,
                                shuffle=False,num_workers=8,pin_memory=True)

        print("显卡是否可用", torch.cuda.is_available())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("所用设备为: ", device)
        self.model.to(device)
        self.spheremodel.to(device)



        # 初始化sphere x,y,z,r
        self.sphere32 = torch.from_numpy(self.sphere32.astype(np.float32)).to(device)





        loss_func = nn.L1Loss(reduction='mean')
        #optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        optimizer = optim.Adam(
            [
                {
                    "params": self.model.parameters(),
                    "lr": self.lr,
                },
                {"params": self.spheremodel.parameters(),
                 "lr": self.lr
                }
            ]
        )

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                   factor=0.7,
                                                   verbose=True,
                                                   min_lr=1e-6,
                                                   threshold=1e-4,
                                                   threshold_mode='abs',
                                                   patience=10)

      #  scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150, T_mult=2000, eta_min=0,verbose=True)

        #scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=3, verbose=False)
        #scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, verbose=False)
        #scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=3000,T_mult=4,verbose=False,eta_min=1e-6)
        #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.7,patience=100,verbose=True,threshold=1e-4,
                                                 #  threshold_mode='abs',cooldown=100) #32768,3000
        #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.5)  # 16384 3000
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,'min',
        #                                            factor=0.7,
        #                                            verbose=True,
        #                                            min_lr=1e-6,
        #                                            threshold=1e-4,
        #                                            threshold_mode='abs',
        #                                            patience=10)
        losslist = []
        print('start train')
        #show model structuer
        # summary(self.model, (0,32))
        # summary(self.spheremodel, (0, 4))

        datalist = list(dataloader)
       # predatalist = datalist[:int(len(datalist)/100)]
        # for index in range(len(datalist)):
        #     datalist[index] = [data.to(device) for data in datalist[index]]

        #预训练
        pretrains = []
        print('预训练开始')
        ####
        for _ in range(5):
            prelr = self.lr
            premodel = self.OverfitSDF(self.H, self.N).to(device)
            prespheremodel = self.SpherelatentNet().to(device)
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


            #prescheduler = lr_scheduler.CosineAnnealingWarmRestarts(preoptimizer, T_0=150, T_mult=2000, eta_min=0,verbose=False)

            prescheduler = lr_scheduler.ReduceLROnPlateau(preoptimizer, 'min',
                                                       factor=0.7,
                                                       verbose=True,
                                                       min_lr=1e-6,
                                                       threshold=1e-4,
                                                       threshold_mode='abs',
                                                       patience=10,cooldown=5)
            #prescheduler = lr_scheduler.MultiStepLR(preoptimizer, milestones=[30, 50,80], gamma=0.3)
            # prescheduler = lr_scheduler.ReduceLROnPlateau(preoptimizer, 'min',
            #                                            factor=0.7,
            #                                            verbose=True,
            #                                            min_lr=1e-6,
            #                                            threshold=1e-4,
            #                                            threshold_mode='abs',
            #                                            patience=5)
            ####
            for _ in range(10):
                epoch_loss = 0
                batch_idx = 0
                random.shuffle(datalist)
                for batch_idx, (x_train,y_train) in enumerate(datalist):
                    x_train,y_train = x_train.to(device),y_train.to(device)

                    spherelatent = prespheremodel(self.sphere32)
                    spherelatent = spherelatent.to(device)
                    sphere_train = torch.cat((self.sphere32[:, :3], spherelatent), dim=1)
                    x_train = torch.cat((sphere_train,computeXtrain(x_train,self.sphere32[:,:3],spherelatent)),dim=0)
                    ####
                    y_train = torch.cat((self.sphere32[:,3],y_train),dim=0)
                    preoptimizer.zero_grad()
                    y_pred = premodel(x_train).squeeze(-1)
                    #sphere_pred = premodel(sphere_train).squeeze(-1)
                    loss = my_loss(y_pred,y_train)
                    epoch_loss += loss.item()
                    loss.backward()
                    preoptimizer.step()
                prescheduler.step(epoch_loss)
            pretrain = {
                'sdfModel': premodel.state_dict(),
                'sphereModel': prespheremodel.state_dict(),
                'scheduler': prescheduler.state_dict(),
                'optimizer': preoptimizer.state_dict(),
                'lr': prelr,
                'loss': epoch_loss/(batch_idx+1)
            }
            print(pretrain['loss'])
            pretrains.append(pretrain)

        pretrains.sort(key=itemgetter('loss'))
        prelosses = [item['loss'] for item in pretrains]
        self.model.load_state_dict(pretrains[0]['sdfModel'])
        self.spheremodel.load_state_dict(pretrains[0]['sphereModel'])
        #self.lr = pretrains[0]['lr']
        optimizer.load_state_dict(pretrains[0]['optimizer'])
        scheduler.load_state_dict(pretrains[0]['scheduler'])
        ####
        e = 10
        #print('预训练loss为:{:.6f} , {:.6f} , {:.6f}'.format(prelosses[0],prelosses[1],prelosses[2]))
        print('正式训练开始')
        while e < self.epochs:
            batch_idx = 0
            time0 = time.time()
            epoch_loss = 0
            epoch_error = 0
            random.shuffle(datalist)
            self.model.train(True)
            for batch_idx, (x_train,y_train) in enumerate(datalist):

                x_train,y_train = x_train.to(device),y_train.to(device)

                spherelatent = self.spheremodel(self.sphere32)
                spherelatent = spherelatent.to(device)
                sphere_train = torch.cat((self.sphere32[:, :3], spherelatent), dim=1)
                x_train = torch.cat((sphere_train, computeXtrain(x_train, self.sphere32[:, :3], spherelatent)), dim=0)
                ####
                y_train = torch.cat((self.sphere32[:, 3], y_train), dim=0)
                optimizer.zero_grad()
                y_pred = self.model(x_train).squeeze(-1)
                #sphere_pred = self.model(sphere_train).squeeze(-1)
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
            if e+1==self.epochs or (e+1)%50==0:
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
        ####
        if not os.path.exists('./results/thingi10k/1_1/' + meshname):
            os.mkdir('./results/thingi10k/1_1/' + meshname)

        outputdir = './results/thingi10k/1_1/' + meshname
        model_file = os.path.join(outputdir,str(spherenum)+'in_'+meshname+'.pth')
        image_file = os.path.join(outputdir,str(spherenum)+'in_'+meshname+'.png')
        genmesh_file = os.path.join(outputdir,str(spherenum)+'in_'+meshname+'.obj')
        # surfaceSampler = gm.PointSampler(sphereSampler.mesh, ratio=0.0, std=0.0)
        # surfacePts = surfaceSampler.sample(100000).astype(np.float32)
        # rspherePts = computeXtrain(torch.from_numpy(surfacePts).cuda(),self.sphere32[:,:3],spherelatent)
        # with torch.no_grad():
        #     Pred = self.model(rspherePts).cpu().numpy()

        #surfaceError = np.mean(np.abs(Pred))
       # print('Surface error: {:.4f}'.format(surfaceError))
        #spherelatent = self.spheremodel(self.sphere32)
        self.trained = True
        state = {
            'net': self.model.state_dict(),
            'sphereNet': self.spheremodel.state_dict(),
            'sphere32': self.sphere32
        }
        torch.save(state, model_file)
        plotTrainResults(losslist,image_file)
       # buildGenMesh(self.model,genmesh_file,self.sphere32,self.spheremodel(self.sphere32))

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
          #  net+=[nn.Linear(N, N*2), nn.LeakyReLU(0.1),nn.Linear(N*2, N), nn.LeakyReLU(0.1)]

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

        def __init__(self, H=1, N=29):
            super().__init__()
            assert (N > 0)
            assert (H > 0)

            # Original paper uses ReLU but I found this lead to dying ReLU issues
            # with negative coordinates. Perhaps was not an issue with original paper's
            # dataset?
            net = [nn.Linear(4, 29)]

            self.model = nn.Sequential(*net)

            #nn.init.normal_(self.model[0].weight,mean=0,std=0.1)

        def forward(self, x):
            output = self.model(x)
            return output




