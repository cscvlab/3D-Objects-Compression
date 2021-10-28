import sys
sys.path.append('C:/Users/cscvlab/Desktop/OverfitShapes/mcdc')
import time

from NeuralImplicit import NeuralImplicit

import os



def train(spherenum,spherefile,outspherefile, meshfile,fuspherenum, num_samples,h,n):
    ##########################################################
    ##########################################################
    mesh = NeuralImplicit(H=h,N=n)
    mesh.encode(spherenum,spherefile,outspherefile, meshfile,fuspherenum, num_samples)
    return mesh




# mesh_folder = "./" #PATH TO FOLDER CONTAINING MESH FILES
# parallelForEachFile(mesh_folder, train, [".obj",".stl"], 4)
def main():
    print("我要开始了")
    # cube = NeuralImplicit()
    # # # cube.encode("pyramid.stl")
    # # # cube.model.to(torch.device("cpu"))
    mesh_dir_path = '/home/cscv/桌面/thingi10k/1/'
    sphere_dir_path = './sphere/thingi10kSphere/insphere'
    #outsphere_dir_path = './sphere/multiThingiSphere/outsphere'
    num_samples = 1000000
    f=0
    spherenum = 128
    fuspherenum = 32
    hiddenSizes = [6]
    nodeSizes = [32]
    meshlist = os.listdir(mesh_dir_path)
    nimeshList = [m[:-6]+'.stl' for m in os.listdir('./thingi10k-weightEncoded')]
    trainedList = [m+'.stl' for m in os.listdir('/home/cscv/桌面/overfit_latent/results/thingi10k/1')]
    for index,m in enumerate(meshlist):

        # if m != '1064655.stl':
        #     continue
        if m in trainedList:
            continue
        if m not in nimeshList:
            print('find a nonmesh')
            continue
        time0 = time.time()
        meshfile = mesh_dir_path + m


        spherefile = sphere_dir_path +str(spherenum)+'/'+ os.path.splitext(m)[0]+'.txt'
        #outspherefile = outsphere_dir_path+str(spherenum)+'/'+os.path.splitext(m)[0]+'.txt'

        for h in hiddenSizes:
            for n in nodeSizes:
                # if (h==10 and n==128)or(h==8 and n==128):
                #     continue
                train(spherenum,spherefile, None, meshfile, fuspherenum,num_samples=num_samples,h=h,n=n)

        # time1 = time.time()
        # # pcFile = pointc_dir_path + os.path.splitext(os.path.split(voxelFile)[1])[0] + \
        # #          '_' + str(num_samples) + '_' + str(mesh.H) + 'x' + str(mesh.N) + '.txt'
        # # PcTest.getPointCloud(mesh, sphereFile, pcFile)
        # fileName = os.path.splitext(os.path.split(voxelFile)[1])[0] + \
        #           '_' + str(num_samples) + '_' + str(mesh.H) + 'x' + str(mesh.N)
        #
        # sdfFile = sdf_path_dir + fileName + '.txt'
        # objFile = obj_path_dir+fileName+'.obj'
        # PcTest.getAllSdf(mesh,sphereFile,sdfFile)
        # MarchingCube.getObj(sdfFile,objFile)
        time1 = time.time()

        msg = '训练时间为：{:.2f}\n'.format(
            (time1 - time0))
        print(msg)

if __name__ == '__main__':
    main()

    # mesh.model.to(torch.device("cpu"))
    # renderer = Renderer(*mesh.renderable())
    # num_frames = 60
    # angle_step = (2 * math.pi) / num_frames
    # curr_angle = 0
    # cam = Camera()
    # frames = []
    # for i in range(num_frames):
    #     curr_angle += angle_step
    #     pos = (math.sqrt(3) * math.cos(curr_angle), math.sqrt(3) * math.sin(curr_angle), 1)
    #     print(pos)
    #     cam.position = pos
    #     cam.direction = dir_vec(pos)
    #     cam.side = (0, 1, 0)
    #     renderer.camera = cam
    #     frames += [renderer.render()]
    # num = 0
    # img_dir = './output/'+os.path.splitext(m)[0]
    # if not os.path.exists(img_dir):
    #     os.mkdir(img_dir)
    # for frame in frames:
    #
    #     framepath = img_dir+'/' + str(num) + '.png'
    #     cv2.imwrite(framepath, frame)
    #     num += 1

# renderer = Renderer(*cube.renderable())

#

#
#
# while (True):
#     for frame in frames:
#         cv2.imshow("img", frame)
#         key = cv2.waitKey(int(2.0/num_frames*1000))
#         if (key == 'q'):
#             exit()