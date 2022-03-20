import os
import cv2
import h5py
import numpy as np
from PIL import Image
from multiprocessing import Pool

# train_flist_name = 'protocol_1/flist_10hz_train.txt'
# train_flist_handler = open(train_flist_name, 'w')
# train_flist = []
# eval_flist_name = 'protocol_1/flist_10hz_eval.txt'
# eval_flist_handler = open(eval_flist_name, 'w')
# eval_flist = []

from matplotlib import cm as mpl_cm, colors as mpl_colors
cm = mpl_cm.get_cmap('jet')
norm_gt = mpl_colors.Normalize()
colors = (cm(norm_gt(np.arange(0, 100)))[:, :3] * 255).astype(np.uint8)

for s in ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']:
#def parse_subject(s):
    print(s)
    vid_base_folder = os.path.join(s, 'Videos')
    part_base_folder = os.path.join(s, 'MySegmentsMat/PartLabels')
    
    vid_fnames = os.listdir(vid_base_folder)
    vid_fnames = sorted(vid_fnames)
    
    vid_fnames = [f for f in vid_fnames if not f.startswith('_')]
    
    action_name = [f[:-4] for f in vid_fnames]
    part_fnames = [f.replace('mp4', 'mat') for f in vid_fnames]
    
    vid_paths = [os.path.join(vid_base_folder, f) for f in vid_fnames]
    part_paths = [os.path.join(part_base_folder, f) for f in part_fnames]
    
#     for vp, pp, action in zip(vid_paths, part_paths, action_name):
    def foo(data):
        vp, pp, action = data
        print("starting", vp, pp, action)
        vpd = cv2.VideoCapture(vp)
        ppd = h5py.File(pp, 'r')
        counter = 0
        success = 1
        while success:
            success, image = vpd.read()
            if not success:
                break
            seg_num = ppd['Feat'].shape[0]
            if counter >= seg_num:
                break
            seg = np.transpose(np.array(ppd[ppd['Feat'][counter][0]]))
            
            cur_image_folder = os.path.join('protocol_1/rgb/{}/{}'.format(s, action))
            if not os.path.exists(cur_image_folder):
                os.makedirs(cur_image_folder, exist_ok=True)
            cur_label_folder = os.path.join('protocol_1/seg/{}/{}'.format(s, action))
            if not os.path.exists(cur_label_folder):
                os.makedirs(cur_label_folder, exist_ok=True)
            
            cur_image_path = os.path.join(cur_image_folder, '{}.png'.format(str(counter).zfill(8)))
            cur_label_path = os.path.join(cur_label_folder, '{}.png'.format(str(counter).zfill(8)))
            
            cv2.imwrite(cur_image_path, image)
            seg_img = Image.fromarray(seg).convert('P')
            seg_img.putpalette(colors)
            seg_img.save(cur_label_path)
            
#             if counter % 5 == 0:
#                 if s in ['S9', 'S11']:
#                     eval_flist_handler.write(cur_image_path + '\n')
#                     eval_flist.append(cur_image_path)
#                 else:
#                     train_flist_handler.write(cur_image_path + '\n')
#                     train_flist.append(cur_image_path)
            counter += 1
        print("ending", vp, pp, action)
    
    processNum = 16
    pool = Pool(processNum)
    args_list = [(vp, pp, action) for vp, pp, action in zip(vid_paths, part_paths, action_name)]
    return_value = pool.map(foo, args_list)
            
# train_flist_handler.close()
# eval_flist_handler.close()


#if __name__ == '__main__':
#    for s in ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']:
#        parse_subject(s)
