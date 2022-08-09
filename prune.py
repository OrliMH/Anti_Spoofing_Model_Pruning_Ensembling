import torch
import numpy as np
import time
from prune_filters import *
from prune_valid import get_prune_acc
from test_color import *
from collections import OrderedDict


def get_channel_index(kernel, num_elimination):
    # get cadidate channel index for pruning

    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)  # kernel.size()=(num_out_channels, num_in_channels, w, h), kernel.view(kernel.size(0), -1)-->(num_out_channels, num_in_channels*w*h)
    # calculate sum of every group of filters (num_out_channels, )    
    vals, args = torch.sort(sum_of_kernel) # ascending order; args are indexes of sorted filters(channels)

    return args[:num_elimination].tolist() # num_elimination有修改过，已经不是原来的channel数了
    #这里返回的是待剪掉的channel的index

def index_remove(tensor, dim, index):
    # tensor 权重， dim待修剪的维度(out_channel, in_channel, k, w), index是待修剪维度的元素的index
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size
    new_size = size_

    select_index = list(set(range(tensor.size(dim))) - set(index)) # 保留下来的index，不修剪的index
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))

    return new_tensor # 即将剪掉后保留下来的权重

def get_state_dict(config):
    out_dir = './models'
    # model_A color
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name) # models/
    initial_checkpoint = config.pretrained_model  
    initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)# models/model_A_color_32/checkpoint/global_min_acer_model.pth
    state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)

    return state_dict

def transform(prune_layer): # 权重另外还保存了一份，两份权重相同，都要做相同的裁剪
    cnt = str(int(prune_layer[20])+1)
    prefix = "module.conv"
    suffix = prune_layer[21:]

    return prefix+cnt+suffix

def bn(k, prune_layer):
    # k:module.encoder.layer0.bn1.weight
    # v.size():torch.Size([64])
    # k:module.encoder.layer0.bn1.bias
    # v.size():torch.Size([64])
    # k:module.encoder.layer0.bn1.running_mean
    # v.size():torch.Size([64])
    # k:module.encoder.layer0.bn1.running_var
    # v.size():torch.Size([64])
    # module.encoder.layer0.conv1.weight     k == prune_layer


    # module.conv1.bn1.weight                k
    # module.conv1.bn1.bias                  k
    # module.conv1.bn1.running_mean          k
    # module.conv1.bn1.running_var           k
    list_ = ['.bn1.weight', '.bn1.bias', '.bn1.running_mean', '.bn1.running_var']
    k_cnt = None
    prune_cnt = None
    if k[12:] in list_:
        k_cnt = int(k[11])
        prune_cnt = int(prune_layer[20]) + 1
    
    if k[:21] == prune_layer[:21] and k[21:] in list_:
        return True
    elif k_cnt == prune_cnt and k[12:] in list_:
        return True
    else:
        return False

def get_channel_index_final(config, prune_layer, num_prune_channel):
    state_dict = get_state_dict(config)
    channel_index = None
    for k, v in state_dict.items():
        # module.encoder.layer0.conv1.weight     k == prune_layer
        # module.conv1.conv1.weight              k
        if k == prune_layer: 
            channel_index = get_channel_index(v, num_prune_channel)
    return channel_index
def get_new_conv(conv, num_prune_channel, dim=0):
    if conv.bias is not None:
        bias = True
    else:
        bias = False
    if dim == 0:
        new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                                   out_channels=int(conv.out_channels - num_prune_channel),# channel_index是待剪掉的channel的index
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation, bias=bias, groups=conv.groups)
    else:
        new_conv = torch.nn.Conv2d(in_channels=int(conv.in_channels - num_prune_channel),
                                   out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation, bias=bias)
    return new_conv
def get_new_norm(norm, num_prune_channel):
    new_norm = torch.nn.BatchNorm2d(num_features=int(norm.num_features - num_prune_channel),
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)
    return new_norm
def get_new_linear(old_linear, remove_channels):
    new = torch.nn.Linear(in_features=old_linear.in_features - remove_channels,
                          out_features=old_linear.out_features, bias=old_linear.bias is not None)
    return new



def prune_weight(config, prune_layer, dim, channel_index):
    state_dict = get_state_dict(config)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # module.encoder.layer0.conv1.weight     k == prune_layer
        # module.conv1.conv1.weight              k
        if k == prune_layer or transform(prune_layer) == k: # 对两份相同的conv裁剪
            print("pruned layer:{}".format(k))
            new_weight = index_remove(v, dim, channel_index)
            new_state_dict[k] = new_weight

        elif bn(k, prune_layer): # 对两份相同的bn裁剪
            # print("pruned layer:{}".format(k))
            new_weight = index_remove(v, dim, channel_index)
            new_state_dict[k] = new_weight
            # print("new_weight.shape:{}".format(new_weight.size()))

        else:
            new_state_dict[k] = v
    return new_state_dict

def prune_affected_weight(simple_prune_affected_layers, dim, channel_index, state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # module.encoder.layer0.conv1.weight     k == prune_layer
        # module.conv1.conv1.weight              k
        if k == simple_prune_affected_layers[0] or transform(simple_prune_affected_layers[0]) == k or k == simple_prune_affected_layers[1] or transform(simple_prune_affected_layers[1]) == k:# 对两份相同的conv裁剪
            # print("pruned layer:{}".format(k))
            new_weight = index_remove(v, dim, channel_index)
            new_state_dict[k] = new_weight
        else:
            new_state_dict[k] = v
    return new_state_dict


def group_prune_weight_net(config, steps, i, layer, channel=None, weight=None, net=None):
    num_prune_channel = None
    if channel is None:
        num_prune_channel = steps[i]
        state_dict = get_state_dict(config)
    else:
        num_prune_channel = channel
        state_dict = weight
    new_state_dict = OrderedDict()
    dim = 0
    channel_index = get_channel_index_final(config, layer, num_prune_channel)
    cnt = 0
    flag = False
    # k:module.encoder.layer1.0.conv2.weight
    # v.size():torch.Size([128, 4, 3, 3])     # k == layer
    # k:module.encoder.layer1.0.bn2.weight
    # v.size():torch.Size([128])              # cnt == 1
    # k:module.encoder.layer1.0.bn2.bias
    # v.size():torch.Size([128])              # cnt == 2
    # k:module.encoder.layer1.0.bn2.running_mean
    # v.size():torch.Size([128])              # cnt == 3
    # k:module.encoder.layer1.0.bn2.running_var
    # v.size():torch.Size([128])              # cnt == 4   dim ^= 1
    # k:module.encoder.layer1.0.bn2.num_batches_tracked
    # v.size():torch.Size([])                 # cnt == 5
    # k:module.encoder.layer1.0.conv3.weight
    # v.size():torch.Size([256, 128, 1, 1])   # cnt == 6
    # print("below is weight")
    for k, v in state_dict.items():
        if k == layer or transform(layer) == k: # conv2.weight
            # print(k)
            # print(v.size())
            cnt = 0
            new_weight = index_remove(v, dim, channel_index)
            # print(new_weight.size())
            new_state_dict[k] = new_weight
            flag = True
        elif flag and cnt == 1 : # bn2.weight
            # print(k)
            # print(v.size())
            new_weight = index_remove(v, dim, channel_index)
            # print(new_weight.size())
            new_state_dict[k] = new_weight
        elif flag and cnt == 2: # bn2.bias
            # print(k)
            # print(v.size())
            new_weight = index_remove(v, dim, channel_index)
            # print(new_weight.size())
            new_state_dict[k] = new_weight
        elif flag and cnt == 3: # bn2.running_mean
            # print(k)
            # print(v.size())
            new_weight = index_remove(v, dim, channel_index)
            # print(new_weight.size())
            new_state_dict[k] = new_weight
        elif flag and cnt == 4: # bn2.running_var
            # print(k)
            # print(v.size())
            new_weight = index_remove(v, dim, channel_index)
            # print(new_weight.size())
            new_state_dict[k] = new_weight
            dim ^= 1
        elif flag and cnt == 6: # conv3.weight    1
            # print(k)
            # print(v.size())
            new_weight = index_remove(v, dim, channel_index)
            # print(new_weight.size())
            new_state_dict[k] = new_weight
            dim ^= 1
        else:
            new_state_dict[k] = v
        cnt += 1
    # print("below is net")
    if net is None:
        net = get_net(config)
    
    # group_prune_layers = ['module.encoder.layer1.0.conv2.weight', 
    # 'module.encoder.layer1.1.conv2.weight', 
    # 'module.encoder.layer2.0.conv2.weight',
    # 'module.encoder.layer2.1.conv2.weight',
    # 'module.encoder.layer3.0.conv2.weight',
    # 'module.encoder.layer3.1.conv2.weight',
    # 'module.encoder.layer4.0.conv2.weight',
    # 'module.encoder.layer4.1.conv2.weight'
    # ]
    # print("layer:{}".format(layer))
    cnt = int(layer[20])+1
    # print("cnt:{}".format(cnt))
    if cnt == 2:
        if layer[22] == '0':
            net.conv2[0].conv2 = get_new_conv(net.conv2[0].conv2, num_prune_channel, 0)
            net.conv2[0].bn2 = get_new_norm(net.conv2[0].bn2, num_prune_channel)
            net.conv2[0].conv3 = get_new_conv(net.conv2[0].conv3, num_prune_channel, 1)
            # print("0")
            # print("net.conv2[0].conv2: {}".format(net.conv2[0].conv2))
        else:
            net.conv2[1].conv2 = get_new_conv(net.conv2[1].conv2, num_prune_channel, 0)
            net.conv2[1].bn2 = get_new_norm(net.conv2[1].bn2, num_prune_channel)
            net.conv2[1].conv3 = get_new_conv(net.conv2[1].conv3, num_prune_channel, 1)
            # print("1")
            # print("net.conv2[1].conv2: {}".format(net.conv2[1].conv2))
    elif cnt == 3:
        if layer[22] == '0':
            net.conv3[0].conv2 = get_new_conv(net.conv3[0].conv2, num_prune_channel, 0)
            net.conv3[0].bn2 = get_new_norm(net.conv3[0].bn2, num_prune_channel)
            net.conv3[0].conv3 = get_new_conv(net.conv3[0].conv3, num_prune_channel, 1)
            # print("0")
            # print("net.conv3[0].conv2: {}".format(net.conv3[0].conv2))
        else:
            net.conv3[1].conv2 = get_new_conv(net.conv3[1].conv2, num_prune_channel, 0)
            net.conv3[1].bn2 = get_new_norm(net.conv3[1].bn2, num_prune_channel)
            net.conv3[1].conv3 = get_new_conv(net.conv3[1].conv3, num_prune_channel, 1)
            # print("1")
            # print("net.conv3[1].conv2: {}".format(net.conv3[1].conv2))
    elif cnt == 4:
        if layer[22] == '0':
            net.conv4[0].conv2 = get_new_conv(net.conv4[0].conv2, num_prune_channel, 0)
            net.conv4[0].bn2 = get_new_norm(net.conv4[0].bn2, num_prune_channel)
            net.conv4[0].conv3 = get_new_conv(net.conv4[0].conv3, num_prune_channel, 1)
            # print("0")
            # print("net.conv4[0].conv2: {}".format(net.conv4[0].conv2))
        else:
            net.conv4[1].conv2 = get_new_conv(net.conv4[1].conv2, num_prune_channel, 0)
            net.conv4[1].bn2 = get_new_norm(net.conv4[1].bn2, num_prune_channel)
            net.conv4[1].conv3 = get_new_conv(net.conv4[1].conv3, num_prune_channel, 1)
            # print("1")
            # print("net.conv4[1].conv2: {}".format(net.conv4[1].conv2))
    else:
        if layer[22] == '0':
            net.conv5[0].conv2 = get_new_conv(net.conv5[0].conv2, num_prune_channel, 0)
            net.conv5[0].bn2 = get_new_norm(net.conv5[0].bn2, num_prune_channel)
            net.conv5[0].conv3 = get_new_conv(net.conv5[0].conv3, num_prune_channel, 1)
            # print("0")
            # print("net.conv5[0].conv2: {}".format(net.conv5[0].conv2))
        else:
            net.conv5[1].conv2 = get_new_conv(net.conv5[1].conv2, num_prune_channel, 0)
            net.conv5[1].bn2 = get_new_norm(net.conv5[1].bn2, num_prune_channel)
            net.conv5[1].conv3 = get_new_conv(net.conv5[1].conv3, num_prune_channel, 1)
            # print("1")
            # print("net.conv5[1].conv2: {}".format(net.conv5[1].conv2))
    return new_state_dict, net


    
def downsample(config, steps, i, layer, channel=None, weight=None, net=None):
    # 对于downsample.0，i-2(se_module.fc2.weight)的outchannel由当前conv的outchannel决定, 
    # i-2的后面只有个bias(i-1)；
    # downsample的outchannel还决定se_module.fc2(i+26)的outchannel，
    # se_module.fc2.bias(i+27)，(layer4除外，layer4没有下一个channel了)下一个layer的0.conv1.weight(i+28)的inchannel
    # 决定0.conv3.weight，0.bn3.weight，0.bn3.bias，0.bn3.bias，0.bn3.running_mean，0.bn3.running_var，0.se_module.fc1.weight、1.conv3.weight，1.bn3.weight，1.bn3.bias，1.bn3.running_mean，1.bn3.running_var，1.se_module.fc1.weight
    # 决定下一个layer的0.downsample.0.weight(i+50) 的inchannel

    

    # conv 0: (i-10)  (i-2)  (i)  (i+18)  (i+26)
    # conv 1: (i-4)  (i+6)  (i+24)  (i+28)(有的layer没有)  (i+50)(有的layer没有)
    # bn: (i-9)  (i-8)  (i-7)  (i-6)  (i-1)  (i+1) (i+2)  (i+3)  (i+4) (i+19) (i+20) (i+21) (i+22)  (i+27)


    # k:module.encoder.layer1.0.conv3.weight      
    # v.size():torch.Size([256, 128, 1, 1])        0.conv3.weight         (i-10)   conv  0
    # k:module.encoder.layer1.0.bn3.weight
    # v.size():torch.Size([256])                   0.bn3.weight           (i-9)    bn
    # k:module.encoder.layer1.0.bn3.bias
    # v.size():torch.Size([256])                   0.bn3.bias             (i-8)    bn
    # k:module.encoder.layer1.0.bn3.running_mean
    # v.size():torch.Size([256])                   0.bn3.running_mean     (i-7)    bn
    # k:module.encoder.layer1.0.bn3.running_var
    # v.size():torch.Size([256])                   0.bn3.running_var      (i-6)    bn
    # k:module.encoder.layer1.0.bn3.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer1.0.se_module.fc1.weight
    # v.size():torch.Size([16, 256, 1, 1])        0.se_module.fc1.weight  (i-4)    conv  1
    # k:module.encoder.layer1.0.se_module.fc1.bias
    # v.size():torch.Size([16])
    # k:module.encoder.layer1.0.se_module.fc2.weight
    # v.size():torch.Size([256, 16, 1, 1])           i-2(se_module.fc2.weight)  (i-2)   conv  0
    # k:module.encoder.layer1.0.se_module.fc2.bias
    # v.size():torch.Size([256])                     i-2的后面只有个bias     (i-1)       bn
    # k:module.encoder.layer1.0.downsample.0.weight
    # v.size():torch.Size([256, 64, 1, 1])                                   (i)        conv  0
    # k:module.encoder.layer1.0.downsample.1.weight
    # v.size():torch.Size([256])                                             (i+1)      bn 
    # k:module.encoder.layer1.0.downsample.1.bias
    # v.size():torch.Size([256])                                             (i+2)      bn
    # k:module.encoder.layer1.0.downsample.1.running_mean
    # v.size():torch.Size([256])                                             (i+3)      bn
    # k:module.encoder.layer1.0.downsample.1.running_var
    # v.size():torch.Size([256])                                             (i+4)      bn
    # k:module.encoder.layer1.0.downsample.1.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer1.1.conv1.weight                                 
    # v.size():torch.Size([128, 256, 1, 1])                                   (i+6)     conv  1
    # k:module.encoder.layer1.1.bn1.weight
    # v.size():torch.Size([128])
    # k:module.encoder.layer1.1.bn1.bias
    # v.size():torch.Size([128])
    # k:module.encoder.layer1.1.bn1.running_mean
    # v.size():torch.Size([128])
    # k:module.encoder.layer1.1.bn1.running_var
    # v.size():torch.Size([128])
    # k:module.encoder.layer1.1.bn1.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer1.1.conv2.weight
    # v.size():torch.Size([128, 4, 3, 3])
    # k:module.encoder.layer1.1.bn2.weight
    # v.size():torch.Size([128])
    # k:module.encoder.layer1.1.bn2.bias
    # v.size():torch.Size([128])  (i+14)  bn
    # k:module.encoder.layer1.1.bn2.running_mean
    # v.size():torch.Size([128])
    # k:module.encoder.layer1.1.bn2.running_var
    # v.size():torch.Size([128])
    # k:module.encoder.layer1.1.bn2.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer1.1.conv3.weight              
    # v.size():torch.Size([256, 128, 1, 1])                      1.conv3.weight      (i+18)   conv  0
    # k:module.encoder.layer1.1.bn3.weight
    # v.size():torch.Size([256])                                 1.conv3.weight      (i+19)    bn
    # k:module.encoder.layer1.1.bn3.bias
    # v.size():torch.Size([256])                                 1.conv3.weight      (i+20)    bn
    # k:module.encoder.layer1.1.bn3.running_mean
    # v.size():torch.Size([256])                                 1.bn3.running_mean  (i+21)    bn
    # k:module.encoder.layer1.1.bn3.running_var
    # v.size():torch.Size([256])                                 1.bn3.running_var   (i+22)    bn
    # k:module.encoder.layer1.1.bn3.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer1.1.se_module.fc1.weight
    # v.size():torch.Size([16, 256, 1, 1])                       1.se_module.fc1.weight   (i+24)    conv  1
    # k:module.encoder.layer1.1.se_module.fc1.bias
    # v.size():torch.Size([16])
    # k:module.encoder.layer1.1.se_module.fc2.weight             
    # v.size():torch.Size([256, 16, 1, 1])                       1.se_module.fc2.weight    (i+26)   conv 0
    # k:module.encoder.layer1.1.se_module.fc2.bias
    # v.size():torch.Size([256])                                 1.se_module.fc2.bias      (i+27)   bn
    # k:module.encoder.layer2.0.conv1.weight
    # v.size():torch.Size([256, 256, 1, 1])                      layer2.0.conv1.weight     (i+28)    conv 1
    # k:module.encoder.layer2.0.bn1.weight
    # v.size():torch.Size([256])
    # k:module.encoder.layer2.0.bn1.bias
    # v.size():torch.Size([256])
    # k:module.encoder.layer2.0.bn1.running_mean
    # v.size():torch.Size([256])
    # k:module.encoder.layer2.0.bn1.running_var
    # v.size():torch.Size([256])
    # k:module.encoder.layer2.0.bn1.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer2.0.conv2.weight
    # v.size():torch.Size([256, 8, 3, 3])
    # k:module.encoder.layer2.0.bn2.weight
    # v.size():torch.Size([256])
    # k:module.encoder.layer2.0.bn2.bias
    # v.size():torch.Size([256])
    # k:module.encoder.layer2.0.bn2.running_mean
    # v.size():torch.Size([256])
    # k:module.encoder.layer2.0.bn2.running_var
    # v.size():torch.Size([256])
    # k:module.encoder.layer2.0.bn2.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer2.0.conv3.weight
    # v.size():torch.Size([512, 256, 1, 1])
    # k:module.encoder.layer2.0.bn3.weight
    # v.size():torch.Size([512])
    # k:module.encoder.layer2.0.bn3.bias
    # v.size():torch.Size([512])
    # k:module.encoder.layer2.0.bn3.running_mean
    # v.size():torch.Size([512])
    # k:module.encoder.layer2.0.bn3.running_var
    # v.size():torch.Size([512])
    # k:module.encoder.layer2.0.bn3.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer2.0.se_module.fc1.weight
    # v.size():torch.Size([32, 512, 1, 1])
    # k:module.encoder.layer2.0.se_module.fc1.bias
    # v.size():torch.Size([32])
    # k:module.encoder.layer2.0.se_module.fc2.weight
    # v.size():torch.Size([512, 32, 1, 1])
    # k:module.encoder.layer2.0.se_module.fc2.bias
    # v.size():torch.Size([512])
    # k:module.encoder.layer2.0.downsample.0.weight
    # v.size():torch.Size([512, 256, 1, 1])                    0.downsample.0.weight   (i+50)     conv  1
    # k:module.encoder.layer2.0.downsample.1.weight
    # v.size():torch.Size([512])
    # k:module.encoder.layer2.0.downsample.1.bias
    # v.size():torch.Size([512])
    # k:module.encoder.layer2.0.downsample.1.running_mean
    # v.size():torch.Size([512])
    # k:module.encoder.layer2.0.downsample.1.running_var
    # v.size():torch.Size([512])
    # k:module.encoder.layer2.0.downsample.1.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer2.1.conv1.weight
    # v.size():torch.Size([256, 512, 1, 1])
    # k:module.encoder.layer2.1.bn1.weight
    # v.size():torch.Size([256])
    # k:module.encoder.layer2.1.bn1.bias
    # v.size():torch.Size([256])
    # k:module.encoder.layer2.1.bn1.running_mean
    # v.size():torch.Size([256])
    # k:module.encoder.layer2.1.bn1.running_var
    # v.size():torch.Size([256])
    # k:module.encoder.layer2.1.bn1.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer2.1.conv2.weight
    # v.size():torch.Size([256, 8, 3, 3])
    # k:module.encoder.layer2.1.bn2.weight
    # v.size():torch.Size([256])
    # k:module.encoder.layer2.1.bn2.bias
    # v.size():torch.Size([256])
    # k:module.encoder.layer2.1.bn2.running_mean
    # v.size():torch.Size([256])
    # k:module.encoder.layer2.1.bn2.running_var
    # v.size():torch.Size([256])
    # k:module.encoder.layer2.1.bn2.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer2.1.conv3.weight
    # v.size():torch.Size([512, 256, 1, 1])
    # k:module.encoder.layer2.1.bn3.weight
    # v.size():torch.Size([512])
    # k:module.encoder.layer2.1.bn3.bias
    # v.size():torch.Size([512])
    # k:module.encoder.layer2.1.bn3.running_mean
    # v.size():torch.Size([512])
    # k:module.encoder.layer2.1.bn3.running_var
    # v.size():torch.Size([512])
    # k:module.encoder.layer2.1.bn3.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer2.1.se_module.fc1.weight
    # v.size():torch.Size([32, 512, 1, 1])
    # k:module.encoder.layer2.1.se_module.fc1.bias
    # v.size():torch.Size([32])
    # k:module.encoder.layer2.1.se_module.fc2.weight
    # v.size():torch.Size([512, 32, 1, 1])
    # k:module.encoder.layer2.1.se_module.fc2.bias
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.0.conv1.weight
    # v.size():torch.Size([512, 512, 1, 1])
    # k:module.encoder.layer3.0.bn1.weight
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.0.bn1.bias
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.0.bn1.running_mean
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.0.bn1.running_var
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.0.bn1.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer3.0.conv2.weight
    # v.size():torch.Size([512, 16, 3, 3])
    # k:module.encoder.layer3.0.bn2.weight
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.0.bn2.bias
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.0.bn2.running_mean
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.0.bn2.running_var
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.0.bn2.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer3.0.conv3.weight
    # v.size():torch.Size([1024, 512, 1, 1])
    # k:module.encoder.layer3.0.bn3.weight
    # v.size():torch.Size([1024])
    # k:module.encoder.layer3.0.bn3.bias
    # v.size():torch.Size([1024])
    # k:module.encoder.layer3.0.bn3.running_mean
    # v.size():torch.Size([1024])
    # k:module.encoder.layer3.0.bn3.running_var
    # v.size():torch.Size([1024])
    # k:module.encoder.layer3.0.bn3.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer3.0.se_module.fc1.weight
    # v.size():torch.Size([64, 1024, 1, 1])
    # k:module.encoder.layer3.0.se_module.fc1.bias
    # v.size():torch.Size([64])
    # k:module.encoder.layer3.0.se_module.fc2.weight
    # v.size():torch.Size([1024, 64, 1, 1])
    # k:module.encoder.layer3.0.se_module.fc2.bias
    # v.size():torch.Size([1024])
    # k:module.encoder.layer3.0.downsample.0.weight
    # v.size():torch.Size([1024, 512, 1, 1])
    # k:module.encoder.layer3.0.downsample.1.weight
    # v.size():torch.Size([1024])
    # k:module.encoder.layer3.0.downsample.1.bias
    # v.size():torch.Size([1024])
    # k:module.encoder.layer3.0.downsample.1.running_mean
    # v.size():torch.Size([1024])
    # k:module.encoder.layer3.0.downsample.1.running_var
    # v.size():torch.Size([1024])
    # k:module.encoder.layer3.0.downsample.1.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer3.1.conv1.weight
    # v.size():torch.Size([512, 1024, 1, 1])
    # k:module.encoder.layer3.1.bn1.weight
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.1.bn1.bias
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.1.bn1.running_mean
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.1.bn1.running_var
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.1.bn1.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer3.1.conv2.weight
    # v.size():torch.Size([512, 16, 3, 3])
    # k:module.encoder.layer3.1.bn2.weight
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.1.bn2.bias
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.1.bn2.running_mean
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.1.bn2.running_var
    # v.size():torch.Size([512])
    # k:module.encoder.layer3.1.bn2.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer3.1.conv3.weight
    # v.size():torch.Size([1024, 512, 1, 1])
    # k:module.encoder.layer3.1.bn3.weight
    # v.size():torch.Size([1024])
    # k:module.encoder.layer3.1.bn3.bias
    # v.size():torch.Size([1024])
    # k:module.encoder.layer3.1.bn3.running_mean
    # v.size():torch.Size([1024])
    # k:module.encoder.layer3.1.bn3.running_var
    # v.size():torch.Size([1024])
    # k:module.encoder.layer3.1.bn3.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer3.1.se_module.fc1.weight
    # v.size():torch.Size([64, 1024, 1, 1])
    # k:module.encoder.layer3.1.se_module.fc1.bias
    # v.size():torch.Size([64])
    # k:module.encoder.layer3.1.se_module.fc2.weight
    # v.size():torch.Size([1024, 64, 1, 1])
    # k:module.encoder.layer3.1.se_module.fc2.bias
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.0.conv1.weight
    # v.size():torch.Size([1024, 1024, 1, 1])
    # k:module.encoder.layer4.0.bn1.weight
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.0.bn1.bias
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.0.bn1.running_mean
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.0.bn1.running_var
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.0.bn1.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer4.0.conv2.weight
    # v.size():torch.Size([1024, 32, 3, 3])
    # k:module.encoder.layer4.0.bn2.weight
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.0.bn2.bias
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.0.bn2.running_mean
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.0.bn2.running_var
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.0.bn2.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer4.0.conv3.weight
    # v.size():torch.Size([2048, 1024, 1, 1])
    # k:module.encoder.layer4.0.bn3.weight
    # v.size():torch.Size([2048])
    # k:module.encoder.layer4.0.bn3.bias
    # v.size():torch.Size([2048])
    # k:module.encoder.layer4.0.bn3.running_mean
    # v.size():torch.Size([2048])
    # k:module.encoder.layer4.0.bn3.running_var
    # v.size():torch.Size([2048])
    # k:module.encoder.layer4.0.bn3.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer4.0.se_module.fc1.weight
    # v.size():torch.Size([128, 2048, 1, 1])
    # k:module.encoder.layer4.0.se_module.fc1.bias
    # v.size():torch.Size([128])
    # k:module.encoder.layer4.0.se_module.fc2.weight
    # v.size():torch.Size([2048, 128, 1, 1])
    # k:module.encoder.layer4.0.se_module.fc2.bias
    # v.size():torch.Size([2048])
    # k:module.encoder.layer4.0.downsample.0.weight
    # v.size():torch.Size([2048, 1024, 1, 1])
    # k:module.encoder.layer4.0.downsample.1.weight
    # v.size():torch.Size([2048])
    # k:module.encoder.layer4.0.downsample.1.bias
    # v.size():torch.Size([2048])
    # k:module.encoder.layer4.0.downsample.1.running_mean
    # v.size():torch.Size([2048])
    # k:module.encoder.layer4.0.downsample.1.running_var
    # v.size():torch.Size([2048])
    # k:module.encoder.layer4.0.downsample.1.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer4.1.conv1.weight
    # v.size():torch.Size([1024, 2048, 1, 1])
    # k:module.encoder.layer4.1.bn1.weight
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.1.bn1.bias
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.1.bn1.running_mean
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.1.bn1.running_var
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.1.bn1.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer4.1.conv2.weight
    # v.size():torch.Size([1024, 32, 3, 3])
    # k:module.encoder.layer4.1.bn2.weight
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.1.bn2.bias
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.1.bn2.running_mean
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.1.bn2.running_var
    # v.size():torch.Size([1024])
    # k:module.encoder.layer4.1.bn2.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer4.1.conv3.weight
    # v.size():torch.Size([2048, 1024, 1, 1])
    # k:module.encoder.layer4.1.bn3.weight
    # v.size():torch.Size([2048])
    # k:module.encoder.layer4.1.bn3.bias
    # v.size():torch.Size([2048])
    # k:module.encoder.layer4.1.bn3.running_mean
    # v.size():torch.Size([2048])
    # k:module.encoder.layer4.1.bn3.running_var
    # v.size():torch.Size([2048])
    # k:module.encoder.layer4.1.bn3.num_batches_tracked
    # v.size():torch.Size([])
    # k:module.encoder.layer4.1.se_module.fc1.weight
    # v.size():torch.Size([128, 2048, 1, 1])
    # k:module.encoder.layer4.1.se_module.fc1.bias
    # v.size():torch.Size([128])
    # k:module.encoder.layer4.1.se_module.fc2.weight
    # v.size():torch.Size([2048, 128, 1, 1])
    # k:module.encoder.layer4.1.se_module.fc2.bias
    # v.size():torch.Size([2048])
    # k:module.encoder.last_linear.weight
    # v.size():torch.Size([1000, 2048])
    # k:module.encoder.last_linear.bias
    # v.size():torch.Size([1000])
    if channel is None:
        num_prune_channel = steps[i]
        state_dict = get_state_dict(config)
    else:
        num_prune_channel = channel
        state_dict = weight
    new_state_dict = OrderedDict()
    channel_index = get_channel_index_final(config, layer, num_prune_channel)


    # print("below is weight")
    target = 0
    target2 = 0
    
    t = 0
    last = 0
    for k, v in state_dict.items(): 
        if k == layer:
            target = t
        elif k == transform(layer):
            target2 = t
        t += 1
        last += 1
    
    t = 0
    last -= 1
    diff = target2 - target
    
    another_last = last - diff
    # print("last:{}".format(last))
    # print("another_last:{}".format(another_last))
    # conv 0: (i-10)  (i-2)  (i)  (i+18)  (i+26)
    # conv 1: (i-4)  (i+6)  (i+24)  (i+28)(有的layer没有)  (i+50)(有的layer没有)
    # bn  0: (i-9)  (i-8)  (i-7)  (i-6)  (i-1)  (i+1) (i+2)  (i+3)  (i+4) (i+19) (i+20) (i+21) (i+22)  (i+27)

    # 特别对最后一个layer，它影响module.id_fc.0.weight(last-1)和module.fc.0.weight(last-3)的in_features， last是最后一个权重，这两项权重在权重文件中没有副本
    # 还有一个可剪可不剪，module.encoder.last_linear.weight，在前向计算时没有用到，但在SENet里定义了，它初始化了参数，就有了权重，它的下标是another_last-3 
    # k:module.encoder.last_linear.weight
    # v.size():torch.Size([1000, 2048])         another_last-3
    # k:module.encoder.last_linear.bias
    # v.size():torch.Size([1000])
    # k:module.conv1.conv1.weight
    # v.size():torch.Size([64, 3, 7, 7])        another_last-1
    # k:module.conv1.bn1.weight
    # v.size():torch.Size([64])                 another_last
    for k, v in state_dict.items():
        # print("t:{}".format(t))
        if t == target-9 or t == target-8 or t == target-7 or t == target-6 or t == target-1 or t == target+1 or t == target+2 or t == target+3 or t == target+4 or t == target+19 or t == target+20 or t == target+21 or t == target+22 or t == target+27 or t == target2-9 or t == target2-8 or t == target2-7 or t == target2-6 or t == target2-1 or t == target2+1 or t == target2+2 or t == target2+3 or t == target2+4 or t == target2+19 or t == target2+20 or t == target2+21 or t == target2+22 or t == target2+27:
            # print(k)
            # print(v.size())
            new_weight = index_remove(v, 0, channel_index)
            # print(new_weight.size())
            new_state_dict[k] = new_weight
        elif t == target-10 or t == target-2 or t == target or t == target+18 or t == target+26 or t == target2-10 or t == target2-2 or t == target2 or t == target2+18 or t == target2+26:
            # print(k)
            # print(v.size())
            new_weight = index_remove(v, 0, channel_index)
            # print(new_weight.size())
            new_state_dict[k] = new_weight
        elif t == target-4 or t == target+6 or t == target+24 or t == target2-4 or t == target2+6 or t == target2+24:
            # print(k)
            # print(v.size())
            new_weight = index_remove(v, 1, channel_index)
            # print(new_weight.size())
            new_state_dict[k] = new_weight
        elif (t == target+28 and layer[20] != '4') or (t == target2+28 and layer[20] != '4') or (t == target+50 and layer[20] != '4') or (t == target2+50 and layer[20] != '4'):
            # print(k)
            # print(v.size())
            new_weight = index_remove(v, 1, channel_index)
            # print(new_weight.size())
            new_state_dict[k] = new_weight
        elif (t == last-1 and layer[20] == '4') or (t == last-3 and layer[20] == '4') or (t == another_last-3 and layer[20] == '4'):
            # print("t, last-1, last-3:")
            # print(k)
            # print(v.size())
            new_weight = index_remove(v, 1, channel_index)
            # print(new_weight.size())
            new_state_dict[k] = new_weight
        else:
            # print(k)
            # print(v.size())
            new_state_dict[k] = v
        t += 1

    # print("below is net")
    if net is None:
        net = get_net(config)
    # module.encoder.layer1.0.downsample.0.weight
    # print("layer:{}".format(layer))
    if layer[20] == '1':
        # k:module.encoder.layer1.0.conv3.weight      
        # v.size():torch.Size([256, 128, 1, 1])        0.conv3.weight         (i-10)   conv  0
        net.conv2[0].conv3 = get_new_conv(net.conv2[0].conv3, num_prune_channel, 0)
        # k:module.encoder.layer1.0.se_module.fc2.weight
        # v.size():torch.Size([256, 16, 1, 1])           i-2(se_module.fc2.weight)  (i-2)   conv  0
        net.conv2[0].se_module.fc2 = get_new_conv(net.conv2[0].se_module.fc2, num_prune_channel, 0)
        # k:module.encoder.layer1.0.downsample.0.weight
        # v.size():torch.Size([256, 64, 1, 1])                                   (i)        conv  0
        net.conv2[0].downsample[0] = get_new_conv(net.conv2[0].downsample[0], num_prune_channel, 0)
        # k:module.encoder.layer1.1.conv3.weight              
        # v.size():torch.Size([256, 128, 1, 1])                      1.conv3.weight      (i+18)   conv  0
        net.conv2[1].conv3 = get_new_conv(net.conv2[1].conv3, num_prune_channel, 0)
        # k:module.encoder.layer1.1.se_module.fc2.weight             
        # v.size():torch.Size([256, 16, 1, 1])                       1.se_module.fc2.weight    (i+26)   conv 0
        net.conv2[1].se_module.fc2 = get_new_conv(net.conv2[1].se_module.fc2, num_prune_channel, 0)

        # k:module.encoder.layer1.0.se_module.fc1.weight
        # v.size():torch.Size([16, 256, 1, 1])        0.se_module.fc1.weight  (i-4)    conv  1
        net.conv2[0].se_module.fc1 = get_new_conv(net.conv2[0].se_module.fc1, num_prune_channel, 1)
         # k:module.encoder.layer1.1.conv1.weight                                 
        # v.size():torch.Size([128, 256, 1, 1])                                   (i+6)     conv  1
        net.conv2[1].conv1 = get_new_conv(net.conv2[1].conv1, num_prune_channel, 1)
        # k:module.encoder.layer1.1.se_module.fc1.weight
        # v.size():torch.Size([16, 256, 1, 1])                       1.se_module.fc1.weight   (i+24)    conv  1
        net.conv2[1].se_module.fc1 = get_new_conv(net.conv2[1].se_module.fc1, num_prune_channel, 1)
        # k:module.encoder.layer2.0.conv1.weight
        # v.size():torch.Size([256, 256, 1, 1])                      layer2.0.conv1.weight     (i+28)    conv 1   conv5没有这个
        net.conv3[0].conv1 = get_new_conv(net.conv3[0].conv1, num_prune_channel, 1)
        # k:module.encoder.layer2.0.downsample.0.weight
        # v.size():torch.Size([512, 256, 1, 1])                    0.downsample.0.weight   (i+50)     conv  1     conv5没有这个
        net.conv3[0].downsample[0] = get_new_conv(net.conv3[0].downsample[0], num_prune_channel, 1)

        # k:module.encoder.layer1.0.bn3.weight
        # v.size():torch.Size([256])                   0.bn3.weight           (i-9)    bn
        # k:module.encoder.layer1.0.bn3.bias
        # v.size():torch.Size([256])                   0.bn3.bias             (i-8)    bn
        # k:module.encoder.layer1.0.bn3.running_mean
        # v.size():torch.Size([256])                   0.bn3.running_mean     (i-7)    bn
        # k:module.encoder.layer1.0.bn3.running_var
        # v.size():torch.Size([256])                   0.bn3.running_var      (i-6)    bn
        net.conv2[0].bn3 = get_new_norm(net.conv2[0].bn3, num_prune_channel)
        # k:module.encoder.layer1.0.se_module.fc2.bias
        # v.size():torch.Size([256])                     i-2的后面只有个bias     (i-1)       bn

        # k:module.encoder.layer1.0.downsample.1.weight
        # v.size():torch.Size([256])                                             (i+1)      bn 
        # k:module.encoder.layer1.0.downsample.1.bias
        # v.size():torch.Size([256])                                             (i+2)      bn
        # k:module.encoder.layer1.0.downsample.1.running_mean
        # v.size():torch.Size([256])                                             (i+3)      bn
        # k:module.encoder.layer1.0.downsample.1.running_var
        # v.size():torch.Size([256])                                             (i+4)      bn
        net.conv2[0].downsample[1] = get_new_norm(net.conv2[0].downsample[1], num_prune_channel)
        # k:module.encoder.layer1.1.bn3.weight
        # v.size():torch.Size([256])                                 1.conv3.weight      (i+19)    bn
        # k:module.encoder.layer1.1.bn3.bias
        # v.size():torch.Size([256])                                 1.conv3.weight      (i+20)    bn
        # k:module.encoder.layer1.1.bn3.running_mean
        # v.size():torch.Size([256])                                 1.bn3.running_mean  (i+21)    bn
        # k:module.encoder.layer1.1.bn3.running_var
        # v.size():torch.Size([256])                                 1.bn3.running_var   (i+22)    bn
        net.conv2[1].bn3 = get_new_norm(net.conv2[1].bn3, num_prune_channel)
        # k:module.encoder.layer1.1.se_module.fc2.bias
        # v.size():torch.Size([256])                                 1.se_module.fc2.bias      (i+27)   bn

    elif layer[20] == '2':
        net.conv3[0].conv3 = get_new_conv(net.conv3[0].conv3, num_prune_channel, 0)
        net.conv3[0].se_module.fc2 = get_new_conv(net.conv3[0].se_module.fc2, num_prune_channel, 0)
        net.conv3[0].downsample[0] = get_new_conv(net.conv3[0].downsample[0], num_prune_channel, 0)
        net.conv3[1].conv3 = get_new_conv(net.conv3[1].conv3, num_prune_channel, 0)
        net.conv3[1].se_module.fc2 = get_new_conv(net.conv3[1].se_module.fc2, num_prune_channel, 0)

    
        net.conv3[0].se_module.fc1 = get_new_conv(net.conv3[0].se_module.fc1, num_prune_channel, 1)
        net.conv3[1].conv1 = get_new_conv(net.conv3[1].conv1, num_prune_channel, 1)
        net.conv3[1].se_module.fc1 = get_new_conv(net.conv3[1].se_module.fc1, num_prune_channel, 1)
        # k:module.encoder.layer2.0.conv1.weight
        # v.size():torch.Size([256, 256, 1, 1])                      layer2.0.conv1.weight     (i+28)    conv 1   conv5没有这个
        net.conv4[0].conv1 = get_new_conv(net.conv4[0].conv1, num_prune_channel, 1)
        net.conv4[0].downsample[0] = get_new_conv(net.conv4[0].downsample[0], num_prune_channel, 1)


        net.conv3[0].bn3 = get_new_norm(net.conv3[0].bn3, num_prune_channel)
        net.conv3[0].downsample[1] = get_new_norm(net.conv3[0].downsample[1], num_prune_channel)
        net.conv3[1].bn3 = get_new_norm(net.conv3[1].bn3, num_prune_channel)
    
    elif layer[20] == '3':
        net.conv4[0].conv3 = get_new_conv(net.conv4[0].conv3, num_prune_channel, 0)
        net.conv4[0].se_module.fc2 = get_new_conv(net.conv4[0].se_module.fc2, num_prune_channel, 0)
        net.conv4[0].downsample[0] = get_new_conv(net.conv4[0].downsample[0], num_prune_channel, 0)
        net.conv4[1].conv3 = get_new_conv(net.conv4[1].conv3, num_prune_channel, 0)
        net.conv4[1].se_module.fc2 = get_new_conv(net.conv4[1].se_module.fc2, num_prune_channel, 0)

    
        net.conv4[0].se_module.fc1 = get_new_conv(net.conv4[0].se_module.fc1, num_prune_channel, 1)
        net.conv4[1].conv1 = get_new_conv(net.conv4[1].conv1, num_prune_channel, 1)
        net.conv4[1].se_module.fc1 = get_new_conv(net.conv4[1].se_module.fc1, num_prune_channel, 1)
        # k:module.encoder.layer2.0.conv1.weight
        # v.size():torch.Size([256, 256, 1, 1])                      layer2.0.conv1.weight     (i+28)    conv 1   conv5没有这个
        net.conv5[0].conv1 = get_new_conv(net.conv5[0].conv1, num_prune_channel, 1)
        net.conv5[0].downsample[0] = get_new_conv(net.conv5[0].downsample[0], num_prune_channel, 1)


        net.conv4[0].bn3 = get_new_norm(net.conv4[0].bn3, num_prune_channel)
        net.conv4[0].downsample[1] = get_new_norm(net.conv4[0].downsample[1], num_prune_channel)
        net.conv4[1].bn3 = get_new_norm(net.conv4[1].bn3, num_prune_channel)
    elif layer[20] == '4':
        net.conv5[0].conv3 = get_new_conv(net.conv5[0].conv3, num_prune_channel, 0)
        net.conv5[0].se_module.fc2 = get_new_conv(net.conv5[0].se_module.fc2, num_prune_channel, 0)
        net.conv5[0].downsample[0] = get_new_conv(net.conv5[0].downsample[0], num_prune_channel, 0)
        net.conv5[1].conv3 = get_new_conv(net.conv5[1].conv3, num_prune_channel, 0)
        net.conv5[1].se_module.fc2 = get_new_conv(net.conv5[1].se_module.fc2, num_prune_channel, 0)

    
        net.conv5[0].se_module.fc1 = get_new_conv(net.conv5[0].se_module.fc1, num_prune_channel, 1)
        net.conv5[1].conv1 = get_new_conv(net.conv5[1].conv1, num_prune_channel, 1)
        net.conv5[1].se_module.fc1 = get_new_conv(net.conv5[1].se_module.fc1, num_prune_channel, 1)



        net.conv5[0].bn3 = get_new_norm(net.conv5[0].bn3, num_prune_channel)
        net.conv5[0].downsample[1] = get_new_norm(net.conv5[0].downsample[1], num_prune_channel)
        net.conv5[1].bn3 = get_new_norm(net.conv5[1].bn3, num_prune_channel)

        # k:module.fc.0.weight
        # v.size():torch.Size([2, 2048])
        # k:module.fc.0.bias
        # v.size():torch.Size([2])
        net.fc[0] = get_new_linear(net.fc[0], num_prune_channel)
        # k:module.id_fc.0.weight
        # v.size():torch.Size([300, 2048])
        # k:module.id_fc.0.bias
        # v.size():torch.Size([300])
        net.id_fc[0] = get_new_linear(net.id_fc[0], num_prune_channel)

        # k:module.encoder.last_linear.weight
        # v.size():torch.Size([1000, 2048])         another_last-3
        net.encoder.last_linear = get_new_linear(net.encoder.last_linear, num_prune_channel)
        
    return new_state_dict, net
    


def resnet_prune_weight_net(config, steps, i, layer, channel, weight, net):


    # if layer[22:] == '0.conv3.weight':
    #     new_state_dict, net = zero_conv3(config, steps, i, layer)
    # elif layer[22:] == '0.downsample.0.weight':
    new_state_dict, net = downsample(config, steps, i, layer, channel, weight, net)
    # else:
    #     new_state_dict, net = one_conv3(config, steps, i, layer)

    return new_state_dict, net
    
def simple_prune_net(config, simple_prune_layers, simple_prune_channels, simple_prune_affected_layers):
    # simple_prune_layers = ['module.encoder.layer0.conv1.weight'] # 同步prune bn1.weight(i+1)，bn1.bias(i+2)，决定紧接着的conv(i+3)的inchannel，
    # #决定module.encoder.layer1.0.downsample.0.weight(i+16)的inchannel
    # simple_prune_channels = [64]
    # 同步prune bn1.weight(i+1)，bn1.bias(i+2)，决定紧接着的conv(i+3)的inchannel，
    # 决定module.encoder.layer1.0.downsample.0.weight(i+16)的inchannel
    # simple_prune_channels有修改过，比原来的数值小
    prune_step_ratio = 1/8
    max_channel_ratio = 0.90 

    simple_prune_channel = simple_prune_channels[0]
    simple_prune_layer = simple_prune_layers[0]

    step = np.linspace(0, int(simple_prune_channel*max_channel_ratio), int(1/prune_step_ratio), dtype=np.int)# 生成(1/prune_step_ratio)个[0, int(simple_prune_channels*max_channel_ratio)]范围内的数，simple_prune_channels是当下的conv的原来的channel个数
    steps = step[1:].tolist() # 剪掉的channels数目的list，由小到多，把0去掉
    
    log = Logger()
    prune_dir_pth = "prune_record"
    if not os.path.exists(prune_dir_pth):
        os.mkdir(prune_dir_pth)
    prune_txt_pth = os.path.join(prune_dir_pth, config.model+"_"+config.image_mode+"_"+config.image_size+"_simple_prune.txt")
    log.open(prune_txt_pth,mode='a')
    log.write("\n")
    
    
    # print("simple prune:")
    log.write(simple_prune_layer)
    log.write("\n")
    log.write("len(steps):{}".format(len(steps)))
    log.write("\n")
    for i in range(len(steps)):
        log.write("i:{}".format(i))
        log.write("\n")
        # print("\n%s: %s Layer, %d Channels pruned"%(time.ctime(), simple_prune_layer, steps[i]))

        net = get_net(config)
 
        # set prune information
        prune_layer = simple_prune_layer
        num_prune_channel = steps[i]

        dim = 0
        net.conv1[0] = get_new_conv(net.conv1[0], num_prune_channel, dim)

        net.conv1[1] = get_new_norm(net.conv1[1], num_prune_channel)

        channel_index = get_channel_index_final(config, prune_layer, num_prune_channel) 
        
        
        weight = prune_weight(config, prune_layer, dim, channel_index)

        dim = 1
        net.conv2[0].conv1 = get_new_conv(net.conv2[0].conv1, num_prune_channel, dim)
       
        net.conv2[0].downsample[0] = get_new_conv(net.conv2[0].downsample[0], num_prune_channel,dim)
        

        weight = prune_affected_weight(simple_prune_affected_layers, dim, channel_index, weight)
 

        net = torch.nn.DataParallel(net)
        net.load_state_dict(weight)

        acc = get_prune_acc(config, net)
        log.write("pruned {} channels {}, acc: {}".format(num_prune_channel, num_prune_channel/simple_prune_channel, acc))
        log.write("\n")
        # print("prune {}, acc {}".format(steps[i], acc))

def group_prune_net(config, group_prune_layers, group_prune_channels, groups):
    log = Logger()
    prune_dir_pth = "prune_record"
    if not os.path.exists(prune_dir_pth):
        os.mkdir(prune_dir_pth)
    prune_txt_pth = os.path.join(prune_dir_pth, config.model+"_"+config.image_mode+"_"+config.image_size+"_group_prune.txt")
    log.open(prune_txt_pth,mode='a')
    log.write("\n")
    for layer, channel in zip(group_prune_layers, group_prune_channels):
        log.write("layer: {}, channel: {}".format(layer, channel))
        log.write("\n")
        steps = [i*groups for i in range(channel//groups)]# 128/32 = 4 [0*32, 1*32, 2*32, 3*32] 3*32/128=3*32/4*32=3/4=.75 len=n
        steps = steps[1:]# len=n-1
        for i in range(len(steps)):
            weight, net = group_prune_weight_net(config, steps, i, layer, None, None, None)
            net = torch.nn.DataParallel(net)
            net.load_state_dict(weight)
            acc = get_prune_acc(config, net)
            log.write("prune {}, acc: {}".format(steps[i]/channel, acc))
            log.write("\n")

def resnet_prune_net(config, resnet_prune_layers, resnet_prune_channels):
    log = Logger()
    prune_dir_pth = "prune_record"
    if not os.path.exists(prune_dir_pth):
        os.mkdir(prune_dir_pth)
    prune_txt_pth = os.path.join(prune_dir_pth, config.model+"_"+config.image_mode+"_"+config.image_size+"_resnet_prune.txt")
    log.open(prune_txt_pth,mode='a')
    log.write("\n")
    for layer, channel in zip(resnet_prune_layers, resnet_prune_channels):
        log.write("layer: {}, channel: {}".format(layer, channel))
        log.write("\n")
        steps = [int(channel*i*0.1) for i in range(10)]
        steps = steps[1:]
        for i in range(len(steps)):
            weight, net = resnet_prune_weight_net(config, steps, i, layer, None, None, None)
            net = torch.nn.DataParallel(net)
            net.load_state_dict(weight)
            acc = get_prune_acc(config, net)
            log.write("prune {} channels {}%, acc: {}".format(steps[i], (i+1)*10, acc))
            log.write("\n")




