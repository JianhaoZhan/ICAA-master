from dataset.dataset import *
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
import numpy as np
from dataset.preprocess_data import *
from PIL import Image, ImageFilter
import argparse
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from models.ablation.model_t import generate_model#all
from opts import parse_opts
from torch.autograd import Variable
import time
import sys
from utils import *
#from utils import AverageMeter, calculate_accuracy
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
if __name__=="__main__":
    opt = parse_opts()
    print(opt)
    
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    # opt.manual_seed = 1, add random seed
    torch.manual_seed(opt.manual_seed)

    print("Preprocessing train data ...")
    train_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 1, opt = opt,test_flag=False)
    print("Length of train data = ", len(train_data))

    print("Preprocessing validation data ...")
    val_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 2, opt = opt,test_flag=False)
    print("Length of validation data = ", len(val_data))
    
    if opt.modality=='RGB': opt.input_channels = 3
    elif opt.modality=='Flow': opt.input_channels = 2

    print("Preparing datatloaders ...")
    train_dataloader = DataLoader(train_data, batch_size = opt.batch_size, shuffle=True, num_workers = opt.n_workers, pin_memory = True, drop_last=True)
    val_dataloader   = DataLoader(val_data, batch_size = opt.batch_size, shuffle=True, num_workers = opt.n_workers, pin_memory = True, drop_last=True)
    print("Length of train datatloader = ",len(train_dataloader))
    print("Length of validation datatloader = ",len(val_dataloader))    
   
    # define the model and load pre-train model
    print("Loading model... ", opt.model, opt.model_depth)
    model, parameters = generate_model(opt)
    
    criterion = nn.CrossEntropyLoss().cuda()

    if opt.resume_path1:
        from models.resnext import get_fine_tuning_parameters
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        print('loading checkpoint {}'.format(opt.resume_path1))
        checkpoint = torch.load(opt.resume_path1)
        assert opt.arch == checkpoint['arch']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], False)
    
    log_path = opt.result_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        
    if opt.log == 1:
        if opt.pretrain_path:
            epoch_logger = Logger(os.path.join(log_path, 'tra.log'),['epoch', 'loss', 'acc', 'lr'], opt.resume_path1, opt.begin_epoch-1)
            val_logger   = Logger(os.path.join(log_path, 'val.log'),['epoch', 'loss', 'acc'], opt.resume_path1, opt.begin_epoch-1)
        else:
            epoch_logger = Logger(os.path.join(log_path, 'tra.log'),['epoch', 'loss', 'acc', 'lr'], opt.resume_path1, opt.begin_epoch-1)
            val_logger   = Logger(os.path.join(log_path, 'val.log'),['epoch', 'loss', 'acc'], opt.resume_path1, opt.begin_epoch-1)
           
    print("Initializing the optimizer ...")
    if opt.pretrain_path: 
        opt.weight_decay = 1e-5
        opt.learning_rate = 0.01

    if opt.nesterov: dampening = 0
    else: dampening = opt.dampening
    
    
    optimizer = optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[9,20], gamma=0.1,  last_epoch=-1)## mean:(9,25),nomean:(6,25)

    if opt.resume_path1 != '':
        # optimizer.load_state_dict(torch.load(opt.resume_path1)['optimizer'])
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10,30], gamma=0.1, last_epoch=-1)  # finetune 4layer
    print('run')
    for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
        
        model.train()
        # model.eval()
        
        print('the learning rate is :{}'.format(optimizer.param_groups[-1]["lr"]))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        for i, (inputs, targets) in enumerate(train_dataloader):
            data_time.update(time.time() - end_time)
        
            targets = targets.cuda(non_blocking=True)
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            # clear grad
            optimizer.zero_grad()


            # BP algorithm
            loss.backward()
            # update parameters
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(train_dataloader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies))
                      
        if opt.log == 1:
            epoch_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'acc': accuracies.avg,
                'lr': optimizer.param_groups[0]['lr']
            })

        
        if epoch % opt.checkpoint == 0 and epoch>9: # and accuracies.avg>0.85:
            if opt.pretrain_path:
                save_file_path = os.path.join(log_path, 'varLR{}.pth'
                            .format(epoch))
            else:
                save_file_path = os.path.join(log_path, 'varLR{}.pth'
                            .format(epoch))
            # save model
            states = {
                'epoch': epoch + 1,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
        # let val do not change weight
        model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        if epoch>7: 
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(val_dataloader):
                    # pdb.set_trace()
                    data_time.update(time.time() - end_time)
                    targets = targets.cuda(non_blocking=True)
                    inputs = Variable(inputs)
                    targets = Variable(targets)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    acc = calculate_accuracy(outputs, targets)
                
                    losses.update(loss.item(), inputs.size(0))
                    accuracies.update(acc, inputs.size(0))

                    batch_time.update(time.time() - end_time)
                    end_time = time.time()

                    print('Val_Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                            epoch,
                            i + 1,
                            len(val_dataloader),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            acc=accuracies))
              
        if opt.log == 1:
            val_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
        # set it to change lr
        scheduler.step()
        



