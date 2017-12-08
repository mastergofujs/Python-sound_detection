import os
import pickle as pkl


# get files'size
def get_FileSize(filePath):
    fsize = os.path.getsize(filePath)
    return fsize


# check if size of file < 1, if true, consider is as empty file
def check_empty_file(path):
    empty_list=list()
    for file in os.listdir(path):
        if get_FileSize(path+file)<1:
            empty_list.append(file)
    return empty_list


path='new_data/negative/1000_500_2/img/'
empty_list=check_empty_file(path)
pkl.dump(empty_list,open('empty_list.pkl','wb'))