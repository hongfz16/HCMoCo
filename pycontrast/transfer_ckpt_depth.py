import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--old_pth', type=str)
parser.add_argument('--new_pth', type=str)
args = parser.parse_args()

old_ckpt = torch.load(args.old_pth, map_location='cpu')

new_dict = {}

# for k, v in old_ckpt['model'].items():
#     if k[7:15] == 'encoder1':
#         new_dict[k.replace('encoder1', 'backbone')] = v
#         print(k.replace('encoder1', 'backbone'))

for k, v in old_ckpt['model'].items():
    if k[7:15] == 'encoder2':
        new_dict[k[16:]] = v
        print(k[16:],end=' ')

torch.save(new_dict, args.new_pth, _use_new_zipfile_serialization=False)
