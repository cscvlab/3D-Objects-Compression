import os
import random
import shutil
def randomFile(dirpath,filenum,outpath):
    randomList = random.sample(range(9800),1000)
    files = os.listdir(dirpath)
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    for i in randomList:
        meshfile = dirpath+files[i]
        shutil.copy(meshfile,outpath+files[i])



if __name__ == '__main__':
    dirpath = '/mnt/hgfs/thingi10k/'
    outpath = '/home/zany2333/桌面/selectData/'
    randomFile(dirpath,1000,outpath)