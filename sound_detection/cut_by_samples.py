import wave
import numpy as np
import struct
pos_path='new_data/positive/'
neg_path='new_data/negative/'
pos_file=pos_path+'new_data_pos2.wav'
neg_file=neg_path+'new_data_neg2.wav'


# cut the sound which cut and merged before manually by sampling and save them as short samples.
def cut_by_samples(file,width,overlap,outpath):
    original_sound = wave.open(file,'r')
    params = original_sound.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    data=original_sound.readframes(nframes)
    data=np.fromstring(data,dtype=np.int16)
    data=data.reshape(-1,2)
    data=data.T
    data=data[0]
    data = data * 1.0 / (max(abs(data)))  # normalization data
    num=1
    original_sound.close()
    for i in range(0,len(data),width-overlap):
        sampBegin=i
        sampEnd=i+width
        subdata=data[sampBegin:sampEnd]
        outname=file.split('_')[2].split('.')[0]+'_'+str(num)+'.wav'
        outwave = wave.open(outpath + outname, 'wb')
        outwave.setparams((1, sampwidth, 48000, len(subdata), "NONE", "not compressed"))
        for v in subdata:
            outwave.writeframes(struct.pack('h', int(v * 48000 / 2)))
        outwave.close()
        num=num+1
        if num % 100 == 0:
            print(i)
            print("cut nums:",num)


# change path to change the sound you want to cut
cut_by_samples(pos_file,500,100,pos_path+'500_100_2/')
cut_by_samples(pos_file,750,200,pos_path+'750_200_2/')
cut_by_samples(pos_file,1000,500,pos_path+'1000_500_2/')

cut_by_samples(neg_file,500,100,neg_path+'500_100_2/')
cut_by_samples(neg_file,750,200,neg_path+'750_200_2/')
cut_by_samples(neg_file,1000,500,neg_path+'1000_500_2/')