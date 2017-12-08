import os
import numpy as np
import wave
import matplotlib
from matplotlib import pylab

matplotlib.use('Agg')
pos_root_path='positive/'
neg_root_path='negative/'


# convert wave file to image file.
def getimg(path):
    time=1
    for file in os.listdir(path):
        if os.path.isfile(path+file):
            try:
                wavefile=wave.open(path+file,'r')
                framerate = wavefile.getframerate()
                numframes = wavefile.getnframes()
                data=wavefile.readframes(numframes)
                data=np.fromstring(data,dtype=np.int16)
                data = data * 1.0 / (max(abs(data)))  # data归一化
                Fs = framerate
                pylab.specgram(data, NFFT=1024, Fs=Fs)
                if not os.path.exists(path+'/img/'+file+'.png'):
                    pylab.savefig(path+'/img/'+file+'.png')
                    pylab.close()
            except :
                print('EOFError happened at ',time,' ',file)
                return 0,0
            time=time+1
            if(time%1==0):
                print('Samples: ',time,' ',file)
    print('Done!')


# training data of negative
path=pos_root_path+'500_100_2/'
getimg(path)
path=pos_root_path+'750_200_2/'
getimg(path)
path=pos_root_path+'1000_500_2/'
getimg(path)


# training data of positive

path=neg_root_path+'500_100_2/'
getimg(path)
path=neg_root_path+'750_200_2/'
getimg(path)
path=neg_root_path+'1000_500_2/'
getimg(path)

