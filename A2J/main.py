import cv2
import torch
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import scipy.io as scio
import os
import model as model
import anchor as anchor
from tqdm import tqdm
import random_erasing
import logging
import time
import random
import h5py
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_pth', type=str, default=None)
parser.add_argument('--output', type=str, default='./result/ITOP')
parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--bs', type=int, default=50)
parser.add_argument('--use_01', action='store_true', default=False)
parser.add_argument('--use_001', action='store_true', default=False)
parser.add_argument('--use_0005', action='store_true', default=False)
parser.add_argument('--use_0002', action='store_true', default=False)
parser.add_argument('--use_0001', action='store_true', default=False)
args = parser.parse_args()

# DataHyperParms 
keypointsNumber = 15
cropWidth = 288
cropHeight = 288
batch_size = 12
depthFactor = 50

learning_rate = 0.00035
Weight_Decay = 1e-4
nepoch = args.bs
RegLossFactor = 3
spatialFactor = 0.5
RandCropShift = 10
RandshiftDepth = 1
RandRotate = 15 
RandScale = (1.0, 0.5)
xy_thres = 110
depth_thres = 150

# randomseed = 75423
# random.seed(randomseed)
# np.random.seed(randomseed)
# torch.manual_seed(randomseed)

save_dir = args.output

try:
    os.makedirs(save_dir)
except OSError:
    pass

trainingImageDir = './data/side_train/'
testingImageDir = './data/side_test/'
# train_keypoint_file = '/data/ITOP/ITOP_side_train_labels.h5'
# test_keypoint_file = '/data/ITOP/ITOP_side_test_labels.h5'
result_file = 'result_ITOP.txt'

Img_mean = np.load('./data/itop_side_mean.npy')[3]
Img_std = np.load('./data/itop_side_std.npy')[3]

bndbox_train = pickle.load(open('./data/bounding_box_depth_train.pkl', 'rb'))
bndbox_test = scio.loadmat('./data/itop_side_bndbox_test.mat' )['FRbndbox_test']

#arch = 'ResNet'
#width = 50
#pretrain_pth = None
#P_h = None
#P_w = None

arch = 'HRNet'
width = 18
pretrain_pth = args.pretrained_pth
P_w = np.array([1,2,3])
P_h = np.array([1,2,3])

def pixel2world(x,y,z):
    worldX = (x - 160.0)*z*0.0035
    worldY = (120.0 - y)*z*0.0035
    return worldX,worldY
    
def world2pixel(x,y,z):
    pixelX = 160.0 + x / (0.0035 * z)
    pixelY = 120.0 - y / (0.0035 * z)
    return pixelX,pixelY
    
joint_id_to_name = {
  0: 'Head',
  1: 'Neck',
  2: 'RShoulder',
  3: 'LShoulder',
  4: 'RElbow',
  5: 'LElbow',
  6: 'RHand',
  7: 'LHand',
  8: 'Torso',
  9: 'RHip',
  10: 'LHip',
  11: 'RKnee',
  12: 'LKnee',
  13: 'RFoot',
  14: 'LFoot',
}

# ## loading GT keypoints and center points
# keypoints_train = h5py.File(train_keypoint_file, 'r')
# keypoints_test = h5py.File(test_keypoint_file, 'r')

def transform(img, label, matrix):
    '''
    img: [H, W]  label, [N,2]   
    '''
    img_out = cv2.warpAffine(img,matrix,(cropWidth,cropHeight))
    label_out = np.ones((keypointsNumber, 3))
    label_out[:,:2] = label[:,:2].copy()
    label_out = np.matmul(matrix, label_out.transpose())
    label_out = label_out.transpose()

    return img_out, label_out

def dataPreprocess(img, keypointsPixel, keypointsWorld, lefttop_pixel, rightbottom_pixel, center, depth_thres=0.4, augment=False):
    
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32')
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32')
    
    if augment:
        RandomOffset_1 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_2 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_3 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_4 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffsetDepth = np.random.normal(0, RandshiftDepth, cropHeight*cropWidth).reshape(cropHeight,cropWidth)
        RandomOffsetDepth[np.where(RandomOffsetDepth < RandshiftDepth)] = 0
        RandomRotate = np.random.randint(-1*RandRotate,RandRotate)
        RandomScale = np.random.rand()*RandScale[0]+RandScale[1]
        matrix = cv2.getRotationMatrix2D((cropWidth/2,cropHeight/2),RandomRotate,RandomScale)
    else:
        RandomOffset_1, RandomOffset_2, RandomOffset_3, RandomOffset_4 = 0, 0, 0, 0
        RandomRotate = 0
        RandomScale = 1
        RandomOffsetDepth = 0
        matrix = cv2.getRotationMatrix2D((cropWidth/2,cropHeight/2),RandomRotate,RandomScale)

    new_Xmin = max(lefttop_pixel[0] + RandomOffset_1, 0)
    new_Ymin = max(lefttop_pixel[1] + RandomOffset_2, 0)
    new_Xmax = min(rightbottom_pixel[0] + RandomOffset_3, img.shape[1] - 1)
    new_Ymax = min(rightbottom_pixel[1] + RandomOffset_4, img.shape[0] - 1)

    # print(new_Ymin, new_Ymax, new_Xmin, new_Xmax)

    imCrop = img[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)].copy()
    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)
    imgResize = np.asarray(imgResize, dtype=np.float32)

    std = 1
    imgResize, mean = crop_human_pcd(imgResize, keypointsWorld[:, 2])
    imgResize[imgResize != 0] = (imgResize[imgResize != 0] - mean) / std

    ## label
    label_xy = np.ones((keypointsNumber, 2), dtype = 'float32') 
    label_xy[:,0] = (keypointsPixel[:,0] - new_Xmin) * cropWidth / (new_Xmax - new_Xmin)
    label_xy[:,1] = (keypointsPixel[:,1] - new_Ymin) * cropHeight / (new_Ymax - new_Ymin)

    if augment:
        imgResize, label_xy = transform(imgResize, label_xy, matrix)

    imageOutputs[:,:,0] = imgResize

    labelOutputs[:,1] = label_xy[:,0]
    labelOutputs[:,0] = label_xy[:,1]
    labelOutputs[:,2] = (keypointsWorld[:,2] - mean) / std * depthFactor
    
    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)

    return data, label, mean, std

def crop_human_pcd(depth, label_z):
    max_z = label_z.max()
    filtered_depth = depth.copy()
    filtered_depth[depth > max_z + 0.05] = 0
    if (filtered_depth != 0).sum() == 0:
        mean = 0
    else:
        mean = filtered_depth.sum() / (filtered_depth != 0).sum()
    return filtered_depth, mean


######################   Pytorch dataloader   #################
class my_dataloader(torch.utils.data.Dataset):

    def __init__(self, ImgDir, bndbox, augment=False):

        self.ImgDir = ImgDir
        self.bndbox = bndbox
        self.length = len(os.listdir(self.ImgDir))
        self.augment = augment
        self.randomErase = random_erasing.RandomErasing(probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0])

        if augment and args.use_01:
            self.length = self.length // 10
            self.index_mapper = {i:i*10 for i in range(self.length)}
        elif augment and args.use_001:
            self.length = self.length // 100
            self.index_mapper = {i:i*100 for i in range(self.length)}
        elif augment and args.use_0001:
            self.length = self.length // 1000
            self.index_mapper = {i:i*1000 for i in range(self.length)}
        elif augment and args.use_0005:
            self.length = self.length // 200
            self.index_mapper = {i:i*200 for i in range(self.length)}
        elif augment and args.use_0002:
            self.length = self.length // 500
            self.index_mapper = {i:i*500 for i in range(self.length)}
        else:
            self.index_mapper = {i:i for i in range(self.length)}

    def __getitem__(self, index):
        index = self.index_mapper[index]

        data = scio.loadmat(self.ImgDir + str(index+1) + '.mat')    
        depth = data['DepthNormal'][:,:,3]
        keypointsPixel = data['keypointsPixel']
        keypointsWorld = data['keypointsWorld']
        
        if isinstance(self.bndbox, dict):
            if self.bndbox[str(index+1)].shape[0] == 0:
                # print("skip")
                return self.__getitem__((index + 1)%self.__len__())
            # print(self.bndbox[str(index+1)])
            lefttop_pixel = self.bndbox[str(index+1)][0, 0:2]
            lefttop_pixel[0] -= 25
            lefttop_pixel[1] -= 15
            rightbottom_pixel = self.bndbox[str(index+1)][0, 2:4]
            rightbottom_pixel[0] += 25
            rightbottom_pixel[1] += 15
        else:
            lefttop_pixel = self.bndbox[index][0:2]
            rightbottom_pixel = self.bndbox[index][2:4]
        data, label, mean, std = dataPreprocess(depth, keypointsPixel, keypointsWorld, lefttop_pixel, rightbottom_pixel, None, augment=self.augment)

        # if self.augment:
        #     data = self.randomErase(data)

        return data, label, torch.from_numpy(keypointsWorld.astype(np.float32)), mean, std
    
    def __len__(self):
        return self.length

      
train_image_datasets = my_dataloader(trainingImageDir, bndbox_train, augment=True)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size = batch_size,
                                             shuffle = True, num_workers = 8)
TrainImgFrames = len(train_image_datasets)
if args.use_01:
    assert TrainImgFrames == 1799
elif args.use_001:
    assert TrainImgFrames == 179
elif args.use_0001:
    assert TrainImgFrames == 17
elif args.use_0005:
    assert TrainImgFrames == 89
elif args.use_0002:
    assert TrainImgFrames == 35
else:
    assert TrainImgFrames == 17991

test_image_datasets = my_dataloader(testingImageDir, bndbox_test, augment=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size,
                                             shuffle = False, num_workers = 8)
TestImgFrames = len(test_image_datasets)
assert TestImgFrames == 4863

def train():
    if arch == 'HRNet':
        net = model.A2J_HRNet_model(num_classes = keypointsNumber,
                                    num_anchors = P_h.shape[0] * P_w.shape[0],
                                    width = width,
                                    pretrain_pth = pretrain_pth,
                                    is_3D = True)
        post_precess = anchor.post_process(shape=[cropHeight//4,cropWidth//4],stride=4,P_h=P_h, P_w=P_w)
        criterion = anchor.A2J_loss(shape=[cropHeight//4,cropWidth//4],thres = [16.0,32.0],stride=4,\
                                    spatialFactor=spatialFactor,img_shape=[cropHeight, cropWidth],P_h=P_h, P_w=P_w)
    elif arch == 'ResNet':
        net = model.A2J_model(num_classes = keypointsNumber)
        post_precess = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None)
        criterion = anchor.A2J_loss(shape=[cropHeight//16,cropWidth//16],thres = [16.0,32.0],stride=16,\
                                    spatialFactor=spatialFactor,img_shape=[cropHeight, cropWidth],P_h=None, P_w=None)
    else:
        raise NotImplementedError
    net = net.cuda()
    
    
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=Weight_Decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
    logging.info('======================================================')

    best_score = -100

    for epoch in range(nepoch):
        net = net.train()
        train_loss_add = 0.0
        Cls_loss_add = 0.0
        Reg_loss_add = 0.0
        timer = time.time()
    
        # Training loop
        for i, (img, label, _, _, _) in enumerate(train_dataloaders):

            torch.cuda.synchronize() 

            img, label = img.cuda(), label.cuda()     
            
            heads  = net(img)  
            #print(regression)     
            optimizer.zero_grad()  
            
            Cls_loss, Reg_loss = criterion(heads, label)

            loss = 1*Cls_loss + Reg_loss*RegLossFactor
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            
            train_loss_add = train_loss_add + (loss.item())*len(img)
            Cls_loss_add = Cls_loss_add + (Cls_loss.item())*len(img)
            Reg_loss_add = Reg_loss_add + (Reg_loss.item())*len(img)

            # printing loss info
            if i%200 == 0:
                print('epoch: ',epoch, ' step: ', i, 'Cls_loss ',Cls_loss.item(), 'Reg_loss ',Reg_loss.item(), ' total loss ',loss.item())

        scheduler.step(epoch)

        # time taken
        torch.cuda.synchronize()
        timer = time.time() - timer
        timer = timer / TrainImgFrames
        print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

        train_loss_add = train_loss_add / TrainImgFrames
        Cls_loss_add = Cls_loss_add / TrainImgFrames
        Reg_loss_add = Reg_loss_add / TrainImgFrames
        print('mean train_loss_add of 1 sample: %f, #train_indexes = %d' %(train_loss_add, TrainImgFrames))
        print('mean Cls_loss_add of 1 sample: %f, #train_indexes = %d' %(Cls_loss_add, TrainImgFrames))
        print('mean Reg_loss_add of 1 sample: %f, #train_indexes = %d' %(Reg_loss_add, TrainImgFrames))


        if (epoch % args.freq == 0) or epoch == nepoch-1: 
            net = net.eval()
            output = torch.FloatTensor()
            all_label = torch.FloatTensor()

            for i, (img, _, label, mean, std) in tqdm(enumerate(test_dataloaders)):
                with torch.no_grad():
                    img = img.cuda()
                    heads = net(img)
                    pred_keypoints = post_precess(heads, voting=False).cpu()
                    pred_keypoints[:, :, -1] = (pred_keypoints[:, :, -1] / float(depthFactor) / std.unsqueeze(-1).float()) + mean.unsqueeze(-1)
                    output = torch.cat([output, pred_keypoints], 0)
                    all_label = torch.cat([all_label, label], 0)

            result = output.cpu().data.numpy()
            Accuracy_test = evaluation10CMRule(result,all_label.numpy())
            evaluation10CMRule_perJoint(result,all_label.numpy())
            if best_score < Accuracy_test:
                best_score = Accuracy_test
                saveNamePrefix = '%s/net_%d_wetD_' % (save_dir, epoch) + str(best_score)
                torch.save(net.state_dict(), saveNamePrefix + '.pth')
            print('epoch: ', epoch, 'Test acc:', Accuracy_test, 'Current Best:', best_score)

        # log
        logging.info('Epoch#%d: total loss=%.4f, Cls_loss=%.4f, Reg_loss=%.4f, Acc_test=%.4f, lr = %.6f'
        %(epoch, train_loss_add, Cls_loss_add, Reg_loss_add, Accuracy_test, scheduler.get_lr()[0]))
        


def test(model_dir=None):   
    net = model.A2J_model(num_classes = keypointsNumber)
    net.load_state_dict(torch.load(model_dir))
    net = net.cuda()
    net.eval()
    
    post_precess = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None)

    output = torch.FloatTensor()
    all_label = torch.FloatTensor()
    torch.cuda.synchronize() 
    for i, (img, _, label, mean, std) in tqdm(enumerate(test_dataloaders)):    
        with torch.no_grad():

            img, label = img.cuda(), label.cuda()    
            heads = net(img)  
            pred_keypoints = post_precess(heads,voting=False).cpu()
            pred_keypoints[:, :, -1] = (pred_keypoints[:, :, -1] / float(depthFactor) / std.unsqueeze(-1).float()) + mean.unsqueeze(-1)
            output = torch.cat([output,pred_keypoints], 0)
            all_label = torch.cat([all_label, label], 0)
        
    torch.cuda.synchronize()       

    result = output.cpu().data.numpy()
    Accuracy_test = evaluation10CMRule(result,all_label.numpy())
    evaluation10CMRule_perJoint(result,all_label.numpy())
    print('Acc:', Accuracy_test)
    

def evaluation10CMRule(source, target):
    assert np.shape(source)==np.shape(target), "source has different shape with target"
    Test1_ = np.zeros(source.shape)
    Test1_[:, :, 0] = source[:,:,1]
    Test1_[:, :, 1] = source[:,:,0]
    Test1_[:, :, 2] = source[:,:,2]
    Test1 = Test1_
    
    for i in range(len(Test1_)):
        Test1[i,:,0] = Test1_[i,:,0]*(bndbox_test[i,2]-bndbox_test[i,0])/cropWidth + bndbox_test[i,0]  # x
        Test1[i,:,1] = Test1_[i,:,1]*(bndbox_test[i,3]-bndbox_test[i,1])/cropHeight + bndbox_test[i,1]  # y   
        # Test1[i,:,2] = Test1_[i,:,2]/depthFactor 
    
    TestWorld = np.ones((len(Test1),keypointsNumber,3))    
    TestWorld_tuple = pixel2world(Test1[:,:,0],Test1[:,:,1],Test1[:,:,2])
    
    TestWorld[:,:,0] = TestWorld_tuple[0]
    TestWorld[:,:,1] = TestWorld_tuple[1]
    TestWorld[:,:,2] = Test1[:,:,2]

    count = 0
    for i in range(len(source)):
        for j in range(keypointsNumber):
            if np.square(TestWorld[i,j,0] - target[i,j,0]) + np.square(TestWorld[i,j,1] - target[i,j,1]) + np.square(TestWorld[i,j,2] - target[i,j,2])<np.square(0.1): #10cm   
                count = count + 1         
    accuracy = count/(len(source)*keypointsNumber)
    return accuracy


def evaluation10CMRule_perJoint(source, target):
    assert np.shape(source)==np.shape(target), "source has different shape with target"
    Test1_ = np.zeros(source.shape)
    Test1_[:, :, 0] = source[:,:,1]
    Test1_[:, :, 1] = source[:,:,0]
    Test1_[:, :, 2] = source[:,:,2]
    Test1 = Test1_  # [x, y, z]
    
    for i in range(len(Test1_)):
        Test1[i,:,0] = Test1_[i,:,0]*(bndbox_test[i,2]-bndbox_test[i,0])/cropWidth + bndbox_test[i,0]  # x
        Test1[i,:,1] = Test1_[i,:,1]*(bndbox_test[i,3]-bndbox_test[i,1])/cropHeight + bndbox_test[i,1]  # y 
        # Test1[i,:,2] = Test1_[i,:,2]/depthFactor 
    TestWorld = np.ones((len(Test1),keypointsNumber,3))    
    TestWorld_tuple = pixel2world(Test1[:,:,0],Test1[:,:,1],Test1[:,:,2])
    
    TestWorld[:,:,0] = TestWorld_tuple[0]
    TestWorld[:,:,1] = TestWorld_tuple[1]
    TestWorld[:,:,2] = Test1[:,:,2]

    count = 0
    accuracy = 0
    for j in range(keypointsNumber):
        for i in range(len(source)):      
            if np.square(TestWorld[i,j,0] - target[i,j,0]) + np.square(TestWorld[i,j,1] - target[i,j,1]) + np.square(TestWorld[i,j,2] - target[i,j,2])<np.square(0.1): #10cm   
                count = count + 1     

        accuracy = count/(len(source))
        print('joint_', j,joint_id_to_name[j], ', accuracy: ', accuracy)
        accuracy = 0
        count = 0

if __name__ == '__main__':
    train()
    # test()
