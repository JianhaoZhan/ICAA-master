from __future__ import division
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math,sys
from functools import partial
import pdb
from torch import Tensor
import numpy as np

__all__ = ['ResNeXt', 'resnet50', 'resnet101']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out




class sta(nn.Module):
  # slowfast idea->r
  def __init__(self,input_shape,r=256):  # r=32 or 256
    super(sta, self).__init__() 
    _,self.c,self.l,self.h,self.w  = input_shape
    self.r = r 
    self.relu = torch.nn.ReLU()
    self.sigmoid = torch.nn.Sigmoid()
    self.serial = True
    print('------------t---------------')
    print('here is att_mul,r is :{},usage of serial is :{}. All in all, Leimu is the NO.1 in the world!'.format(r,self.serial))
    print('------------t---------------')
    self.At1 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(self.c,1),1.0/(self.c))), requires_grad=True)# input:0.1~,0.1*I*c=0.1
    self.At2 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(self.h*self.w,1),0.5/(self.h*self.w))), requires_grad=True)# input:0.2~
    self.Et1 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(self.r*self.l, self.l),1.0/self.l)), requires_grad=True)# I*0.1*l=0.1
    self.Et2 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(self.r*self.l, self.l),1.0/self.l)), requires_grad=True)
    self.St1 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(self.l, self.r*self.l),1.0/(self.r*self.l))), requires_grad=True)# I*0.1*r*l=0.1
    self.St2 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(self.l, self.r*self.l),1.0/(self.r*self.l))), requires_grad=True)
    # self.Ac1 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(self.l,1),1.0/self.l)), requires_grad=True)# input:0.1~,0.1*I*l=0.1
    # self.Ac2 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(self.h*self.w,1),0.5/(self.h*self.w))), requires_grad=True)# input:0.2~
    # self.Ec1 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(int(self.c/self.r),self.c),1.0/self.c)), requires_grad=True)# I*0.1*c=0.1
    # self.Ec2 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(int(self.c/self.r),self.c),1.0/self.c)), requires_grad=True)
    # self.Sc1 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(self.c,int(self.c/self.r)),1.0/(self.c/self.r))), requires_grad=True)# I*0.1*c/r=0.01
    # self.Sc2 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(self.c,int(self.c/self.r)),1.0/(self.c/self.r))), requires_grad=True)
    # self.As1 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(self.l,1),1.0/self.l)), requires_grad=True)# input:0.1~,0.1*I*l=0.1
    # self.As2 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(self.c,1),0.5/self.c)), requires_grad=True)# input:0.1~,0.2*I*c=0.1
    # self.Es1 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(self.r,self.h*self.w),1.0/(self.h*self.w))), requires_grad=True)
    # self.Es2 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(self.r,self.h*self.w),1.0/(self.h*self.w))), requires_grad=True)
    # self.Ss1 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(self.h*self.w,self.r),1.0/(self.r))), requires_grad=True)
    # self.Ss2 = torch.nn.Parameter(torch.FloatTensor(nn.init.constant_(torch.Tensor(self.h*self.w,self.r),1.0/(self.r))), requires_grad=True)
    
  def forward(self,sta_input: Tensor)-> Tensor:
    self.b = int(sta_input.shape[0]) # input : (b,c,l,h,w)
    if not self.serial:
        # T -> T1 + T2 + T3   
        input_t = sta_input.transpose(1,2).reshape([self.b,self.l,self.c,self.h*self.w])#(b,l,c,hw)
        # T1 (b,l,c,hw)->(b,l,c){*At1(c, 1)}->{Et1(rl,l)*}(b,l,1)->{St1(l,rl)*}(b,rl,1)->(b,l,1)
        input_t1 = torch.mean(input_t, 3, False)#(b,l,c)
        input_t1 = torch.matmul(input_t1,self.At1)# torch.div(torch.matmul(input_t1,self.At1),self.c)
        input_t1 = self.relu(torch.matmul(self.Et1,input_t1))# self.relu(torch.div(torch.matmul(self.Et1,input_t1),self.l)) 
        input_t1 = self.sigmoid(torch.matmul(self.St1,input_t1))# self.sigmoid(torch.div(torch.matmul(self.St1,input_t1),self.r*self.l))
        # T2 (b,l,c,hw)->(b,l,hw){*At2(hw,1)}->{Et2(rl,l)*}(b,l,1)->{St2(l,rl)*}(b,rl,1)->(b,l,1)
        input_t2 = torch.mean(input_t, 2, False)#(b,l,hw)
        input_t2 = torch.matmul(input_t2,self.At2)# torch.div(torch.matmul(input_t2,self.At2),self.h*self.w)
        input_t2 = self.relu(torch.matmul(self.Et2,input_t2))# self.relu(torch.div(torch.matmul(self.Et2,input_t2),self.l))
        input_t2 = self.sigmoid(torch.matmul(self.St2,input_t2))# self.sigmoid(torch.div(torch.matmul(self.St2,input_t2),self.r*self.l))
        # add
        # input_t = torch.add(input_t1,input_t2,input_t3)
        # multiply
        input_t = torch.add(input_t1,input_t2)
        input_t = torch.div(input_t,(input_t.mean()))
        # (b,l,1)->(b,1,l,1,1)
        input_t = input_t.unsqueeze(1).unsqueeze(-1)
        if input_t.shape != (self.b, 1, self.l, 1, 1):
            raise ValueError('the shape of temporal data is error !')
            
        # # C -> C1 + C2 + C3
        # input_c = sta_input.reshape([self.b,self.c,self.l,self.h*self.w])#(b,c,l,hw)
        # # C1 (b,c,l,hw)->(b,c,l){*Ac1(l,1)}-> {Ec1(c/r,c)*}(b,c,1)->{Sc1(c,c/r)*}(b,c/r,1)->(b,c,1)
        # input_c1 = torch.mean(input_c, 3, False)#(b,c,l)
        # input_c1 = torch.matmul(input_c1,self.Ac1)# torch.div(torch.matmul(input_c1,self.Ac1),self.l)
        # input_c1 = self.relu(torch.matmul(self.Ec1,input_c1)) #self.relu(torch.div(torch.matmul(self.Ec1,input_c1),self.c))
        # input_c1 = self.sigmoid(torch.matmul(self.Sc1,input_c1))# self.sigmoid(torch.div(torch.matmul(self.Sc1,input_c1),int(self.c/self.r)))
        # # C2 (b,c,l,hw)->(b,c,hw){*Ac2(hw,1)}->{Ec2(c/r,c)*}(b,c,1)->{Sc2(c,c/r)*}(b,c/r,1)->(b,c,1)
        # input_c2 = torch.mean(input_c, 2, False)#(b,c,hw)
        # input_c2 = torch.matmul(input_c2,self.Ac2)# torch.div(torch.matmul(input_c2,self.Ac2),self.h*self.w)
        # input_c2 = self.relu(torch.matmul(self.Ec2,input_c2))# self.relu(torch.div(torch.matmul(self.Ec2,input_c2),self.c))
        # input_c2 = self.sigmoid(torch.matmul(self.Sc2,input_c2))# self.sigmoid(torch.div(torch.matmul(self.Sc2,input_c2),int(self.c/self.r)))
        # # add
        # # input_c = torch.add(input_c1,input_c2,input_c3)
        # # multiply
        # input_c = torch.add(input_c1,input_c2)
        # input_c = torch.div(input_c,(input_c.mean()))
        # # (b,c,1)->(b,c,1,1,1)
        # input_c = input_c.unsqueeze(-1).unsqueeze(-1)
        # if input_c.shape != (self.b, self.c, 1, 1, 1):
        #     raise ValueError('the shape of channel data is error !')
            
        # # S -> S1 + S2 + S3
        # input_s =  sta_input.reshape([self.b,self.c,self.l,self.h*self.w]).transpose(1,3)#(b,c,l,hw)->(b,hw,l,c)
        # # S1 (b,hw,l,c)->(b,hw,l)->(b,hw,l){*As1(l,1)}->{Es1(lhw,hw)*}(b,hw,1)->{Ss1(hw,lhw)*}(b,lhw,1)->(b,1,1,h,w)
        # input_s1 = torch.mean(input_s, 3, False)#(b,hw,l)
        # input_s1 = torch.matmul(input_s1,self.As1)# torch.div(torch.matmul(input_s1,self.As1),self.l)
        # input_s1 = self.relu(torch.matmul(self.Es1,input_s1))# self.relu(torch.div(torch.matmul(self.Es1,input_s1),self.h*self.w))
        # input_s1 = self.sigmoid(torch.matmul(self.Ss1,input_s1))# self.sigmoid(torch.div(torch.matmul(self.Ss1,input_s1),self.l*self.h*self.w))
        # # S2 (b,hw,l,c)->(b,hw,c)->(b,hw,c){*As2(c,1)}->{Es2(lhw,hw)*}(b,hw,1)->{Ss2(hw,lhw)*}(b,lhw,1)->(b,1,1,h,w)
        # input_s2 = torch.mean(input_s, 2, False)#(b,hw,c)
        # input_s2 = torch.matmul(input_s2,self.As2)# torch.div(torch.matmul(input_s2,self.As2),self.c)
        # input_s2 = self.relu(torch.matmul(self.Es2,input_s2)) #self.relu(torch.div(torch.matmul(self.Es2,input_s2),self.h*self.w))
        # input_s2 = self.sigmoid(torch.matmul(self.Ss2,input_s2))# self.sigmoid(torch.div(torch.matmul(self.Ss2,input_s2),self.l*self.h*self.w))
        # # add
        # # input_s = torch.add(input_s1,input_s2,input_s3)
        # # multiply
        # input_s = torch.add(input_s1,input_s2)#.reshape([self.b,1,1,self.h,self.w]) # (b,1,1,h,w)
        # input_s = torch.div(input_s,(input_s.mean())).reshape([self.b,1,1,self.h,self.w])
        # if input_s.shape != (self.b, 1, 1, self.h, self.w):
        #     raise ValueError('the shape of spatial data is error !')

        sta_input = torch.mul(sta_input,torch.mul(torch.mul(input_t,input_c),input_s))
        return sta_input
    else:
        # T -> T1 + T2 + T3   
        input_t = sta_input.transpose(1,2).reshape([self.b,self.l,self.c,self.h*self.w])#(b,l,c,hw)
        # T1 (b,l,c,hw)->(b,l,c){*At1(c, 1)}->{Et1(rl,l)*}(b,l,1)->{St1(l,rl)*}(b,rl,1)->(b,l,1)
        input_t1 = torch.mean(input_t, 3, False)#(b,l,c)
        input_t1 = torch.matmul(input_t1,self.At1)# torch.div(torch.matmul(input_t1,self.At1),self.c)
        input_t1 = self.relu(torch.matmul(self.Et1,input_t1))# self.relu(torch.div(torch.matmul(self.Et1,input_t1),self.l)) 
        input_t1 = self.sigmoid(torch.matmul(self.St1,input_t1))# self.sigmoid(torch.div(torch.matmul(self.St1,input_t1),self.r*self.l))
        # T2 (b,l,c,hw)->(b,l,hw){*At2(hw,1)}->{Et2(rl,l)*}(b,l,1)->{St2(l,rl)*}(b,rl,1)->(b,l,1)
        input_t2 = torch.mean(input_t, 2, False)#(b,l,hw)
        input_t2 = torch.matmul(input_t2,self.At2)# torch.div(torch.matmul(input_t2,self.At2),self.h*self.w)
        input_t2 = self.relu(torch.matmul(self.Et2,input_t2))# self.relu(torch.div(torch.matmul(self.Et2,input_t2),self.l))
        input_t2 = self.sigmoid(torch.matmul(self.St2,input_t2))# self.sigmoid(torch.div(torch.matmul(self.St2,input_t2),self.r*self.l))
        # add
        # input_t = torch.add(input_t1,input_t2,input_t3)
        # multiply
        input_t = torch.add(input_t1,input_t2)
        input_t = torch.div(input_t,torch.mean(input_t))
        # print('t',input_t)
        # (b,l,1)->(b,1,l,1,1)
        input_t = input_t.unsqueeze(1).unsqueeze(-1)
        if input_t.shape != (self.b, 1, self.l, 1, 1):
            raise ValueError('the shape of temporal data is error !')
        sta_input = torch.mul(sta_input, input_t)

        # # C -> C1 + C2 + C3
        # input_c = sta_input.reshape([self.b,self.c,self.l,self.h*self.w])#(b,c,l,hw)
        # # C1 (b,c,l,hw)->(b,c,l){*Ac1(l,1)}-> {Ec1(c/r,c)*}(b,c,1)->{Sc1(c,c/r)*}(b,c/r,1)->(b,c,1)
        # input_c1 = torch.mean(input_c, 3, False)#(b,c,l)
        # input_c1 = torch.matmul(input_c1,self.Ac1)# torch.div(torch.matmul(input_c1,self.Ac1),self.l)
        # input_c1 = self.relu(torch.matmul(self.Ec1,input_c1)) #self.relu(torch.div(torch.matmul(self.Ec1,input_c1),self.c))
        # input_c1 = self.sigmoid(torch.matmul(self.Sc1,input_c1))# self.sigmoid(torch.div(torch.matmul(self.Sc1,input_c1),int(self.c/self.r)))
        # # C2 (b,c,l,hw)->(b,c,hw){*Ac2(hw,1)}->{Ec2(c/r,c)*}(b,c,1)->{Sc2(c,c/r)*}(b,c/r,1)->(b,c,1)
        # input_c2 = torch.mean(input_c, 2, False)#(b,c,hw)
        # input_c2 = torch.matmul(input_c2,self.Ac2)# torch.div(torch.matmul(input_c2,self.Ac2),self.h*self.w)
        # input_c2 = self.relu(torch.matmul(self.Ec2,input_c2))# self.relu(torch.div(torch.matmul(self.Ec2,input_c2),self.c))
        # input_c2 = self.sigmoid(torch.matmul(self.Sc2,input_c2))# self.sigmoid(torch.div(torch.matmul(self.Sc2,input_c2),int(self.c/self.r)))
        # # add
        # # input_c = torch.add(input_c1,input_c2,input_c3)
        # # multiply
        # input_c = torch.add(input_c1,input_c2)
        # input_c = torch.div(input_c,torch.mean(input_c))
        # # print('c',input_c)
        # # (b,c,1)->(b,c,1,1,1)
        # input_c = input_c.unsqueeze(-1).unsqueeze(-1)
        # if input_c.shape != (self.b, self.c, 1, 1, 1):
        #     raise ValueError('the shape of channel data is error !')
        # sta_input = torch.mul(sta_input, input_c)
                
        # # S -> S1 + S2 + S3
        # input_s =  sta_input.reshape([self.b,self.c,self.l,self.h*self.w]).transpose(1,3)#(b,c,l,hw)->(b,hw,l,c)
        # # S1 (b,hw,l,c)->(b,hw,l)->(b,hw,l){*As1(l,1)}->{Es1(lhw,hw)*}(b,hw,1)->{Ss1(hw,lhw)*}(b,lhw,1)->(b,1,1,h,w)
        # input_s1 = torch.mean(input_s, 3, False)#(b,hw,l)
        # input_s1 = torch.matmul(input_s1,self.As1)# torch.div(torch.matmul(input_s1,self.As1),self.l)
        # input_s1 = self.relu(torch.matmul(self.Es1,input_s1))# self.relu(torch.div(torch.matmul(self.Es1,input_s1),self.h*self.w))
        # input_s1 = self.sigmoid(torch.matmul(self.Ss1,input_s1))# self.sigmoid(torch.div(torch.matmul(self.Ss1,input_s1),self.l*self.h*self.w))
        # # S2 (b,hw,l,c)->(b,hw,c)->(b,hw,c){*As2(c,1)}->{Es2(lhw,hw)*}(b,hw,1)->{Ss2(hw,lhw)*}(b,lhw,1)->(b,1,1,h,w)
        # input_s2 = torch.mean(input_s, 2, False)#(b,hw,c)
        # input_s2 = torch.matmul(input_s2,self.As2)# torch.div(torch.matmul(input_s2,self.As2),self.c)
        # input_s2 = self.relu(torch.matmul(self.Es2,input_s2)) #self.relu(torch.div(torch.matmul(self.Es2,input_s2),self.h*self.w))
        # input_s2 = self.sigmoid(torch.matmul(self.Ss2,input_s2))# self.sigmoid(torch.div(torch.matmul(self.Ss2,input_s2),self.l*self.h*self.w))
        # # add
        # # input_s = torch.add(input_s1,input_s2,input_s3)
        # # multiply
        # input_s = torch.add(input_s1,input_s2)#.reshape([self.b,1,1,self.h,self.w]) # (b,1,1,h,w)
        # input_s = torch.div(input_s,torch.mean(input_s)).reshape([self.b,1,1,self.h,self.w])
        # # print('s',input_s,input_s.shape)#,exit()
        # if input_s.shape != (self.b, 1, 1, self.h, self.w):
        #     raise ValueError('the shape of spatial data is error !')
        # sta_input = torch.mul(sta_input, input_s)

        return sta_input



class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None,attention=False,r=256,sta_shape=(3, 2048, 4, 7, 7)):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.attention = attention
        self.sta_shape = sta_shape
        self.r = r
        if attention:
            self.sta = sta(input_shape=self.sta_shape,r=self.r)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.attention==True:
            out = self.sta(out).cuda()

        # downsample = nn.Sequential(nn.Conv3d(self.inplanes,planes * block.expansion,kernel_size=1,
        #                stride=stride,bias=False), nn.BatchNorm3d(planes * block.expansion))
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


''' resnext101: 
           (ResNeXtBottleneck, 
            [3, 4, 23, 3], 
            num_classes=opt.n_classes:101 or 51,
            shortcut_type=opt.resnet_shortcut:B,
            cardinality=opt.resnext_cardinality:32,
            sample_size=opt.sample_size:112,
            sample_duration=opt.sample_duration:16,
            input_channels=opt.input_channels: rgb:2,flow:2,
            output_layers=opt.output_layers: mars:avg_pooling,else:[])
'''
class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400,
                 input_channels=3,
                 output_layers=[]):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            input_channels,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        
        self.bn1 = nn.BatchNorm3d(64) # 64 : num of features
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        # 3 blocks
        self.layer1 = self._make_layer(
            block, 128, layers[0], shortcut_type, cardinality)
        # 4 blocks
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)#, attention=True,r=32,sta_shape=(1, 512, 16, 28, 28))
        # 23 blocks
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        # 3 blocks
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2, attention=True,r=256,sta_shape=(1, 2048, 4, 4, 4))
        # math.ceil : xiang shang qu zheng->1 
        last_duration = int(math.ceil(sample_duration / 16))
        # 224/32=7
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        print('avg_pooling kernal size is :({},{},{}) and it should be (4,7,7)!!!!!'.format(last_duration,last_size,last_size))
        # 32*32*2=2048
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        
        #layer to output on forward pass
        self.output_layers = output_layers

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    # block:ResNeXtBottleneck, planes:128,256,512,1024(num of feature maps in bottleneck structure), 
    # blocks:layers[0-3](resnext101:layer:[3, 4, 23, 3]), shortcut_type:B, cardinality:32,stride=1or2
    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1,
                    attention=False,
                    r=16,
                    sta_shape=(3, 2048, 4, 7, 7)):
        downsample = None
        # block.expansion=2
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                # nn.Sequential : container to connect each modules into a model
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample,attention=attention,r=r,sta_shape=sta_shape))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality,attention=attention,r=r,sta_shape=sta_shape))

        # * let seperate list into element
        return nn.Sequential(*layers)

    def forward(self, x):
        #pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # kernal (1,4,4)
        x5 = self.avgpool(x4)

        # function: transform tensor (b,f,c,h,w) to (b,f*c*h*w)
        x6 = x5.view(x5.size(0), -1)
        x7 = self.fc(x6)
        if len(self.output_layers) == 0:
            return x7
        else:
            out = []
            out.append(x7)
            for i in self.output_layers:
                if i == 'avgpool':
                    out.append(x6)
                if i == 'layer4':
                    out.append(x4)
                if i == 'layer3':
                    out.append(x3)
        return out

    def freeze_batch_norm(self):
        for name,m in self.named_modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d): # PHIL: i Think we can write just  "if isinstance(m, nn._BatchNorm)
                m.eval() # use mean/variance from the training
                m.weight.requires_grad = False
                m.bias.requires_grad = False



def get_fine_tuning_parameters(model, ft_begin_index):# ft_begin_index:4
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')
    ft_module_names.append('layer2')

    print("Layers to finetune is : !!!!!", ft_module_names)

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})
    return parameters


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model
