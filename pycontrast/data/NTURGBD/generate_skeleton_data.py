import os
import pickle
import numpy as np

fname = './NTURGBD/nturgbd_flist_clear.txt'
lines = open(fname, 'r').readlines()
tags = [l.split('/')[-2] for l in lines]
unique_tags = sorted(list(set(tags)))
skeleton_folder = './NTURGBD/nturgb+d_skeletons'
skeleton_flist = [os.path.join(skeleton_folder, t + '.skeleton') for t in unique_tags]
target_folder = './NTURGBD/nturgb+d_parsed_skeleton'

def func(arg):
    t, sf = arg
    with open(sf, 'r') as fd:
        data = fd.readlines()
    joint_data = []
    frame_data = []
    for frame_idx in range(int(data.pop(0))):
        cur_frame_data = {}
        cur_frame_data['frame_idx'] = frame_idx
        cur_frame_data['joints'] = []
        for body_idx in range(int(data.pop(0))):
            body = data.pop(0)
            cur_frame_data['joints'].append({
                'body_idx': body_idx,
                '3d_loc': [],
                'rgb_loc': [],
                'd_loc': [],
            })
            for joint_idx in range(int(data.pop(0))):
                line = data.pop(0).split()
                joint_data.append((frame_idx, body_idx, joint_idx, line[:7]))
                x = np.array(line[:7], dtype=np.float32)
                cur_frame_data['joints'][-1]['3d_loc'].append(list(x[:3]))
                cur_frame_data['joints'][-1]['rgb_loc'].append(list(x[5:7]))
                cur_frame_data['joints'][-1]['d_loc'].append(list(x[3:5]))
        frame_data.append(cur_frame_data)
    cur_target_folder = os.path.join(target_folder, t)
    if not os.path.exists(cur_target_folder):
        os.makedirs(cur_target_folder, exist_ok=True)
    for i, fd in enumerate(frame_data):
        cur_target_fname = os.path.join(cur_target_folder, 'Skeleton-{}.pkl'.format(str(i).zfill(8)))
        with open(cur_target_fname, 'wb') as f:
            pickle.dump(fd, f)
    print(t)
    
from multiprocessing import Pool
processNum = 16
pool = Pool(processNum)
args = [(a,b) for a,b in zip(unique_tags, skeleton_flist)]
pool.map(func, args)
