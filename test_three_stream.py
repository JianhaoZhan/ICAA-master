from dataset.dataset import *
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
import numpy as np
from dataset.preprocess_data import *
import torch
import models.final.model as att#models.serial_model.model_zjh as att # serial_model
import models.final.model_org as org
from opts import parse_opts
from torch.autograd import Variable
import torch.nn.functional as F
import time
import sys
from utils import AverageMeter, calculate_accuracy, calculate_accuracy_video
import random
import pdb
''''
Script to test_three_stream.py
CUDA_VISIBLE_DEVICES=0 python test_three_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 --log 0 --dataset HMDB51 --modality RGB_Flow --sample_duration 64 --split 1   --frame_dir "dataset/HMDB51/" --annotation_path "dataset/HMDB51_labels" --result_path "results/" --resume_path1 --resume_path3 --resume_path2 
CUDA_VISIBLE_DEVICES=0 python test_three_stream.py --batch_size 1 --n_classes 101 --model resnext --model_depth 101 --log 0 --dataset UCF101 --modality RGB_Flow --sample_duration 64 --split 2 --frame_dir "dataset/UCF101" --annotation_path "dataset/UCF101_labels/" --result_path "results/"   --resume_path1  --resume_path3  --resume_path2 
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def test():
    opt = parse_opts()
    print(opt)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    print("Preprocessing validation data ...")
    data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 0, opt = opt)
    print("Length of validation data = ", len(data))

    print("Preparing datatloaders ...")
    val_dataloader = DataLoader(data, batch_size = 1, shuffle=False, num_workers = opt.n_workers, pin_memory = True, drop_last=False)
    print("Length of validation datatloader = ",len(val_dataloader))

    result_path = "{}/{}/".format(opt.result_path, opt.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # define the model
    print("Loading models... ", opt.model, opt.model_depth)
    model1, parameters1 = att.generate_model(opt)   # RGB
    model2, parameters2 = org.generate_model(opt)   # MARS

    # if testing RGB+Flow streams change input channels
    if not opt.only_RGB:
        opt.input_channels = 2
    model3, parameters3 = att.generate_model(opt)   # Flow
   
    if opt.resume_path1:  # RGB
        print('loading checkpoint {}'.format(opt.resume_path1))
        checkpoint = torch.load(opt.resume_path1)
        assert opt.arch == checkpoint['arch']
        model1.load_state_dict(checkpoint['state_dict'],False)
    if opt.resume_path2:  # MARS
        print('loading checkpoint {}'.format(opt.resume_path2))
        checkpoint = torch.load(opt.resume_path2)
        assert opt.arch == checkpoint['arch']
        model2.load_state_dict(checkpoint['state_dict'],False)
    if opt.resume_path3:  # Flow
        print('loading checkpoint {}'.format(opt.resume_path3))
        checkpoint = torch.load(opt.resume_path3)
        assert opt.arch == checkpoint['arch']
        model3.load_state_dict(checkpoint['state_dict'],False)

    model1.eval()
    model2.eval()
    model3.eval()
   
    accuracies = AverageMeter()
   
    if opt.log:
        if opt.only_RGB:
            f = open(os.path.join(root_dir, "test_RGB_MARS_{}{}_{}_{}_{}.txt".format(opt.model, opt.model_depth, opt.dataset, opt.split, opt.sample_duration)), 'w+')
        else:
             f = open(os.path.join(root_dir, "test_RGB_MARS_Flow_{}{}_{}_{}_{}.txt".format(opt.model, opt.model_depth, opt.dataset, opt.split, opt.sample_duration)), 'w+')
        f.write(str(opt))
        f.write('\n')
        f.flush()
   
    with torch.no_grad():
        for i, (clip, label) in enumerate(val_dataloader):
            clip = torch.squeeze(clip)
            if opt.only_RGB:
                inputs = torch.Tensor(int(clip.shape[1]/opt.sample_duration), 3, opt.sample_duration, opt.sample_size, opt.sample_size)
                for k in range(inputs.shape[0]):
                    inputs[k,:,:,:,:] = clip[:,k*opt.sample_duration:(k+1)*opt.sample_duration,:,:]  

                inputs_var1 = Variable(inputs)
                inputs_var2 = Variable(inputs)
            else:
                RGB_clip  = clip[0:3,:,:,:]
                Flow_clip = clip[3:,:,:,:]
                inputs1 = torch.Tensor(int(RGB_clip.shape[1]/opt.sample_duration), 3, opt.sample_duration, opt.sample_size, opt.sample_size)
                inputs2 = torch.Tensor(int(Flow_clip.shape[1]/opt.sample_duration), 2, opt.sample_duration, opt.sample_size, opt.sample_size)
                for k in range(inputs1.shape[0]):
                    inputs1[k,:,:,:,:] = RGB_clip[:,k*opt.sample_duration:(k+1)*opt.sample_duration,:,:]  
                    inputs2[k,:,:,:,:] = Flow_clip[:,k*opt.sample_duration:(k+1)*opt.sample_duration,:,:]
                inputs_var1 = Variable(inputs1)
                inputs_var2 = Variable(inputs2)  
           

            outputs_var1= model1(inputs_var1)   # RGB
            outputs_var2= model2(inputs_var1)   # MARS
            outputs_var3= model3(inputs_var2)   # Flow
               
            outputs_var = torch.mean(torch.cat((outputs_var1, outputs_var2, outputs_var3), dim=0), dim=0).unsqueeze(0)

            pred5 = np.array(outputs_var.topk(5, 1, True)[1].cpu().data[0])
               
            acc = float(pred5[0] == label[0])
               
            accuracies.update(acc, 1)

            line = "Video[" + str(i) + "] : \t top5 " + str(pred5) + "\t top1 = " + str(pred5[0]) +  "\t true = " +str(label[0]) + "\t video = " + str(accuracies.avg)
            print(line)
            if opt.log:
                f.write(line + '\n')
                f.flush()
   
    print("Video accuracy = ", accuracies.avg)
    line = "Video accuracy = " + str(accuracies.avg) + '\n'  
   
if __name__=="__main__":
    test()
