
import os
import numpy as np
import pickle as pkl
from skimage import io,transform
import random
import math


# read and reshape the samples' images and save their gray values as features, meanwhile, give each of them a label
def get_img_features(path,label):
    img_feature=list()
    label_list=list()
    time=1
    for file in os.listdir(path):
        if os.path.isfile(path+file):
            try:
                img=io.imread(path+file,as_grey=True)
                img=transform.resize(img,(48,64))
                img=img[6:43,8:58]
                img=(img-img.min())/(img.max()-img.min())
            except EOFError:
                print('EOFError happened at ',time,' ',file)
                return 0,0
            feature=np.array(img)
            img_feature.append(feature)
            label_list.append(label)
            time=time+1
            if(time%100==0):
                print('Samples: ',time,' ',file)
    print('Done!')
    return img_feature,label_list


# data of positive
path='new_data/positive/1000_500/img/'
data,label=get_img_features(path,np.array([1,0],dtype=int))
pos_data=list(zip(data,label))
random.shuffle(pos_data)
data[:],label[:]=zip(*pos_data)
pos_data_train=tuple((data[0:math.ceil(len(data)*0.8)],label[0:math.ceil(len(data)*0.8)]))
pos_data_test=tuple((data[math.ceil(len(data)*0.8):len(data)],label[math.ceil(len(data)*0.8):len(data)]))

# data of negative
path='new_data/negative/1000_500/img/'
data,label=get_img_features(path,np.array([0,1],dtype=int))
neg_data=list(zip(data,label))
random.shuffle(neg_data)
data[:],label[:]=zip(*neg_data)
neg_data_train=tuple((data[0:math.ceil(len(data)*0.8)],label[0:math.ceil(len(data)*0.8)]))
neg_data_test=tuple((data[math.ceil(len(data)*0.8):len(data)],label[math.ceil(len(data)*0.8):len(data)]))

# training data
train_data=neg_data_train[0]+pos_data_train[0]
train_label=neg_data_train[1]+pos_data_train[1]
train=list(zip(train_data,train_label))
random.shuffle(train)
train_data[:],train_label[:]=zip(*train)
img_features_train=tuple((train_data,train_label))
pkl.dump(img_features_train,open('new_data/new_data_pkl/train_1000_500.pkl','wb'))


# test data
test_data=neg_data_test[0]+pos_data_test[0]
test_label=neg_data_test[1]+pos_data_test[1]
test=list(zip(test_data,test_label))
random.shuffle(test)
test_data[:],test_label[:]=zip(*test)
img_features_test=tuple((test_data,test_label))
pkl.dump(img_features_test,open('new_data/new_data_pkl/test_1000_500.pkl','wb'))
