
import numpy as np
import matplotlib.image as mpimg
import h5py
import os
import scipy.io as scio
import matplotlib.pyplot as plt
import math
import time

split='test'

depth_maps = h5py.File(f'./data/ITOP_side_{split}_depth_map.h5', 'r')
labels = h5py.File(f'./data/ITOP_side_{split}_labels.h5', 'r')
   
saveDir = f'./data/side_{split}/'
def GetDepthNormal(depth_maps,labels):
    DepthNormal = np.zeros((240,320,4),dtype='float32')
    count = 0
    for i in range(depth_maps['data'].shape[0]):
        if labels['is_valid'][i]:
            if count%1000 == 0:
                print(count)
            depth_map = depth_maps['data'][i].astype(np.float32)
            coor_joints = labels['image_coordinates'][i]
            world_joints = labels['real_world_coordinates'][i]
            height, width = np.shape(depth_map)

            # DepthNormal[1:height-1, 1:width-1, 0] = -(depth_map[2:height, 1:width-1] - depth_map[0:height-2, 1:width-1]) / 2.0
            # DepthNormal[1:height-1, 1:width-1, 1] = -(depth_map[1:height-1, 2:width] - depth_map[1:height-1, 0:width-2]) / 2.0
            # DepthNormal[1:height-1, 1:width-1, 2] = 1
            DepthNormal[:,:,3] = depth_map[:,:]            

            # for x in range(1,height-1):
            #     for y in range(1,width-1):
            #         dzdx = (depth_map[x+1,y] - depth_map[x-1,y]) / 2.0
            #         dzdy = (depth_map[x,y+1] - depth_map[x,y-1]) / 2.0
        
            #         # DepthNormal[x,y,0] = -dzdx
            #         # DepthNormal[x,y,1] = -dzdy
            #         # DepthNormal[x,y,2] = 1
            #         assert DepthNormal[x,y,0] == -dzdx
            #         assert DepthNormal[x,y,1] == -dzdy

            count = count+1
        
            scio.savemat(saveDir + str(count) + '.mat', {
                'DepthNormal': DepthNormal,
                'keypointsPixel': coor_joints,
                'keypointsWorld': world_joints})
    print(count)

    return 0
 


if __name__ == '__main__':
    GetDepthNormal(depth_maps,labels)
    print(depth_maps['data'].shape)
    valid = labels['is_valid']
    print(np.array(valid).sum())
    

