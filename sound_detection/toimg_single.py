import os
import numpy as np
import wave
from matplotlib import pylab
import pickle as pkl

pos_root_path='positive/'
neg_root_path='negative/'


# convert single wave file which is empty (because some files maybe wrong when cut by samples) to image
def getimg(path,file):
    time=1
    if os.path.isfile(path+file):
        try:
            wavefile=wave.open(path+file,'r')
            framerate = wavefile.getframerate()
            numframes = wavefile.getnframes()
            data=wavefile.readframes(numframes)
            data=np.fromstring(data,dtype=np.int16)
            data = data * 1.0 / (max(abs(data)))  # data归一化
            Fs = framerate
            pylab.axis('off')
            pylab.specgram(data, NFFT=1024, Fs=Fs)
            # if not os.path.exists(path+'/img/'+file+'.png'):
            pylab.savefig(path+'/img/'+file+'.png')
            pylab.close()
        except :
            print('EOFError happened at ',time,' ',file)
            return 0,0
        time=time+1
        if(time%1==0):
             print('Samples:',file)
    print('Done!')


i=0
empty_list=pkl.load(open('empty_list.pkl','rb'))
path=neg_root_path+'1000_500_2/'
for item in empty_list:
    i=i+1
    file=item.split('.')[0]+'.'+item.split('.')[1]
    getimg(path,file)
    print(i)
