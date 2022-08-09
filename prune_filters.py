#coding:utf-8
import os
# os.environ['CUDA_VISIBLE_DEVICES'] =  '4,5,6,7' #'3,2,1,0'
import sys

from torch.nn.modules import module
sys.path.append("..")
sys.path.append("./process")
sys.path.append("./model")
import torch.nn as nn
import torch

import argparse
from process.data import *
from process.augmentation import *
from metric import *
from loss.cyclic_lr import CosineAnnealingLR_with_Restart

from model.backbone import FaceBagNet  
import sys
import torch
import prune


# 获取不同的模型
def get_model(model_name, num_class,is_first_bn):
    if model_name == 'baseline':
        from model.model_baseline import Net
    elif model_name == 'model_A':
        from model.FaceBagNet_model_A import Net
    elif model_name == 'model_B':
        from model.FaceBagNet_model_B import Net
    elif model_name == 'model_C':
        from model.FaceBagNet_model_C import Net

    net = Net(num_class=num_class,is_first_bn=is_first_bn)
    return net

def get_net(config):
    out_dir = './models'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir, config.model_name)

    net = get_model(model_name=config.model, num_class=2, is_first_bn=True)
    net = net.cpu()

    return net

def get_net_weight(config):
    print("origin_net:")
    out_dir = './models'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir, config.model_name)
    initial_checkpoint = config.pretrained_model


    model = get_model(model_name=config.model, num_class=2, is_first_bn=True)
    model = nn.DataParallel(model)

    if initial_checkpoint is not None:
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    

    return model 

def run_prune(config):
    simple_prune_layers = ['module.encoder.layer0.conv1.weight'] # 同步prune bn1.weight(i+1)，bn1.bias(i+2)，决定紧接着的conv(i+3)的inchannel，
    #决定module.encoder.layer1.0.downsample.0.weight(i+16)的inchannel
    simple_prune_channels = [64]

    simple_prune_affected_layers = ['module.encoder.layer1.0.conv1.weight', 'module.encoder.layer1.0.downsample.0.weight']
    group_prune_layers = ['module.encoder.layer1.0.conv2.weight', 
    'module.encoder.layer1.1.conv2.weight', 
    'module.encoder.layer2.0.conv2.weight',
    'module.encoder.layer2.1.conv2.weight',
    'module.encoder.layer3.0.conv2.weight',
    'module.encoder.layer3.1.conv2.weight',
    'module.encoder.layer4.0.conv2.weight',
    'module.encoder.layer4.1.conv2.weight'
    ] # 当前conv是组卷积，将inchannel分成group组，那么只对outchannel进行prune，prune后outchannel是32倍数(group参数是32)，不对inchannel进行prune，因为inchannel分组后参数是4，8，16,32，参数量少，而且如果inchannel和outchannel同时prune，参数量容易算重复，不好统计，代码也不好实现
    group_prune_channels = [128, 128, 256, 256, 512, 512, 1024, 1024] # 是组卷积的outchannel
    resnet_prune_layers = [ 'module.encoder.layer1.0.downsample.0.weight', 
    'module.encoder.layer2.0.downsample.0.weight',
    'module.encoder.layer3.0.downsample.0.weight', 
    'module.encoder.layer4.0.downsample.0.weight'
    ] 
    resnet_prune_channels = [256, 
    512, 
    1024, 
    2048]

    prune.simple_prune_net(config, simple_prune_layers, simple_prune_channels, simple_prune_affected_layers) # 如果将字典和列表变量传入函数，在函数内部修改字典和列表变量的值，从函数出来后变量会被更改。
    prune.group_prune_net(config, group_prune_layers, group_prune_channels, groups=32)
    prune.resnet_prune_net(config, resnet_prune_layers, resnet_prune_channels)    
    
def main(config):
    if config.mode == 'infer_test':
        config.pretrained_model = r'global_min_acer_model.pth'
        run_prune(config)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default = -1)

    parser.add_argument('--model', type=str, default='model_A')
    parser.add_argument('--image_mode', type=str, default='color')
    parser.add_argument('--image_size', type=int, default=32)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cycle_num', type=int, default=10)
    parser.add_argument('--cycle_inter', type=int, default=50)

    parser.add_argument('--mode', type=str, default='infer_test', choices=['train','infer_test'])
    parser.add_argument('--pretrained_model', type=str, default=None)

    config = parser.parse_args()
    print(config)
    main(config)
