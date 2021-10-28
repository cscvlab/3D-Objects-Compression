import os

if __name__ == '__main__':
    path = '/mnt/hgfs/Thingi10K/raw_meshes/'
    for m in os.listdir(path):
        h = os.path.splitext(m)[1]
        oldname = os.path.splitext(m)[0]
        newname = oldname+'_sf'
        os.rename(path+m,path+newname+h)