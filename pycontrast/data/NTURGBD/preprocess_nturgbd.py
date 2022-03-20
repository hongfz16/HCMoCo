import cv2
import os
import re
import numpy as np
from multiprocessing import Pool
import pickle

root_folder = './NTURGBD'
rgb_folder = os.path.join(root_folder, './nturgb+d_rgb')
depth_folder = os.path.join(root_folder, './nturgb+d_depth_masked')
skeleton_folder = os.path.join(root_folder, './nturgb+d_skeletons')

tags = os.listdir(rgb_folder)
tags = [f.split('_')[0] for f in tags]

video_set = []
compiled_regex = re.compile('.*S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3}).*')
for t in tags:
    match = re.match(compiled_regex, t)
    setup, camera, performer, replication, action = [*map(int, match.groups())]
    video_set.append((setup, camera))

def process_video_set(target_video_set):
    print("Starting {}".format(target_video_set))
    rgb = []
    d = []
    for i in range(len(tags)):
        if video_set[i] != target_video_set or np.random.rand() > 0.5:
            continue
        with open(os.path.join(skeleton_folder, tags[i] + '.skeleton'), 'r') as fd:
            data = fd.readlines()
        joint_data = []
        for frame_idx in range(int(data.pop(0))):
            for body_idx in range(int(data.pop(0))):
                body = data.pop(0)
                for joint_idx in range(int(data.pop(0))):
                    line = data.pop(0).split()
                    if body_idx == 0:
                        joint_data.append((frame_idx, body_idx, joint_idx, line[:7]))
        depth = []
        color = []
        for joint in joint_data:
            x = np.array(joint[3], dtype=np.float32)
            depth.append(x[3:5])
            color.append(x[5:7])
        if len(depth) == 0:
            assert len(color) == 0
            continue
        d.append(np.stack(depth))
        rgb.append(np.stack(color))
    rgb = np.concatenate(rgb).astype(np.float32)
    d = np.concatenate(d).astype(np.float32)
    H, _ = cv2.findHomography(rgb, d, cv2.RANSAC)
    print("Finishing {}".format(target_video_set))
    return (target_video_set, H)

def process_tag(arg):
    tag = arg[0]
    H = arg[1]
    print("Starting {}".format(tag))
    target_folder = os.path.join(root_folder, './nturgb+d_rgb_warped_correction', tag)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)
    vidcap = cv2.VideoCapture(os.path.join(rgb_folder, tag + '_rgb.avi'))
    counter = 1
    success = 1
    while success:
        success, image = vidcap.read()
        if not success:
            break
        warped_image = cv2.warpPerspective(image, H, (512, 424))
        save_image_fname = os.path.join(target_folder, 'WRGB-{}.jpg'.format(str(counter).zfill(8)))
        cv2.imwrite(save_image_fname, warped_image)
        counter += 1
    print("Finishing {} with {} frames".format(tag, counter))

if __name__ == '__main__':
    unique_video_set = set(video_set)
    processNum = 16
    pool = Pool(processNum)
    print("Calculating the Homography...")
    return_value = pool.map(process_video_set, list(unique_video_set))
    homography_dict = {d[0]: d[1] for d in return_value}
    pickle.dump(homography_dict, open('homography_dict_correction.pkl', 'wb'))
    print("Warping RGB...")
    print("Before tags num: {}".format(len(tags)))
    exclude_list = open('./NTU_RGBD_samples_with_missing_skeletons.txt', 'r').readlines()
    exclude_list = [l.strip() for l in exclude_list]
    new_tags = [t for t in tags if t not in exclude_list]
    print("After tags num: {}".format(len(new_tags)))
    homography_dict = pickle.load(open('homography_dict_correction.pkl', 'rb'))
    args = []
    for k in homography_dict.keys():
        ind = video_set.index(k)
        args.append((new_tags[ind], homography_dict[k]))
    pool.map(process_tag, args)
