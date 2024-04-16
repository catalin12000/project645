import read_bvh
import numpy as np
from os import listdir
import os


def generate_pos_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    if (os.path.exists(tar_traindata_folder)==False):
        os.makedirs(tar_traindata_folder)
    bvh_dances_names=listdir(src_bvh_folder)
    for bvh_dance_name in bvh_dances_names:
        name_len=len(bvh_dance_name)
        if(name_len>4):
            if(bvh_dance_name[name_len-4: name_len]==".bvh"):
                dance=read_bvh.get_train_data(src_bvh_folder+bvh_dance_name)
                np.save(tar_traindata_folder+bvh_dance_name+".npy", dance)

def generate_pos_bvh_from_traindata(src_train_folder, tar_bvh_folder):
    if (os.path.exists(tar_bvh_folder)==False):
        os.makedirs(tar_bvh_folder)
    dances_names=listdir(src_train_folder)
    for dance_name in dances_names:
        name_len=len(dance_name)
        if(name_len>4):
            if(dance_name[name_len-4: name_len]==".npy"):
                dance=np.load(src_train_folder+dance_name)
                dance2=[]
                for i in range(int(dance.shape[0]/8)):
                    dance2=dance2+[dance[i*8]]
                read_bvh.write_traindata_to_bvh(tar_bvh_folder+dance_name+".bvh", np.array(dance2))


# Encode data from bvh to positional encoding
generate_pos_traindata_from_bvh("train_data_bvh/martial/","train_data_xyz/martial/")

# Decode from positional to bvh
generate_pos_bvh_from_traindata("train_data_xyz/martial/", "test_data_xyz_bvh/martial/",)
