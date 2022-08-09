from prune import *
from prune_filters import *
from prune_valid import *
from re_train_model_A_color_32 import *


def simple_prune_final(config, net, simple_prune_layers, simple_prune_channels, simple_prune_affected_layers):
    
 
    # set prune information
    prune_layer = simple_prune_layers[0]
    num_prune_channel = simple_prune_channels[0]


    dim = 0
    net.conv1[0] = get_new_conv(net.conv1[0], num_prune_channel, dim)

    net.conv1[1] = get_new_norm(net.conv1[1], num_prune_channel)

    channel_index = get_channel_index_final(config, prune_layer, num_prune_channel) 
    
    
    weight = prune_weight(config, prune_layer, dim, channel_index)

    dim = 1
    net.conv2[0].conv1 = get_new_conv(net.conv2[0].conv1, num_prune_channel, dim)
    
    net.conv2[0].downsample[0] = get_new_conv(net.conv2[0].downsample[0], num_prune_channel,dim)
    

    weight = prune_affected_weight(simple_prune_affected_layers, dim, channel_index, weight)
    return weight, net
 
def group_prune_final(config, weight, net, group_prune_layers, group_prune_channels, resnet_prune_layers):
    for layer, channel in zip(group_prune_layers, group_prune_channels):
        weight, net = group_prune_weight_net(config, None, None, layer, channel, weight, net)
    return weight, net

def resnet_prune_final(config, weight, net, resnet_prune_layers, resnet_prune_channels):
    for layer, channel in zip(resnet_prune_layers, resnet_prune_channels):
        weight, net = downsample(config, None, None, layer, channel, weight, net)
    return weight, net
def final_prune(config):
    simple_prune_layers = ['module.encoder.layer0.conv1.weight'] # 同步prune bn1.weight(i+1)，bn1.bias(i+2)，决定紧接着的conv(i+3)的inchannel，
    #决定module.encoder.layer1.0.downsample.0.weight(i+16)的inchannel
    simple_prune_channels = [32]

    simple_prune_affected_layers = ['module.encoder.layer1.0.conv1.weight', 'module.encoder.layer1.0.downsample.0.weight']
    # layer: module.encoder.layer2.1.conv2.weight, channel: 256
    # prune 0.875, acc: 0.9435886761032473
    # layer: module.encoder.layer3.0.conv2.weight, channel: 512
    # prune 0.5, acc: 0.9567027477102414
    # layer: module.encoder.layer3.1.conv2.weight, channel: 512
    # prune 0.9375, acc: 0.9484804329725229
    # layer: module.encoder.layer4.0.conv2.weight, channel: 1024
    # prune 0.96875, acc: 0.9574313072439634
    # layer: module.encoder.layer4.1.conv2.weight, channel: 1024
    # prune 0.96875, acc: 0.9661740216486261
    group_prune_layers = [ 
    'module.encoder.layer3.1.conv2.weight',
    'module.encoder.layer4.0.conv2.weight',
    'module.encoder.layer4.1.conv2.weight'
    ] # 当前conv是组卷积，将inchannel分成group组，那么只对outchannel进行prune，prune后outchannel是32倍数(group参数是32)，不对inchannel进行prune，因为inchannel分组后参数是4，8，16,32，参数量少，而且如果inchannel和outchannel同时prune，参数量容易算重复，不好统计，代码也不好实现
    group_prune_channels = [int(512*0.0625), int(1024*0.625), int(1024*0.96875)] # 是组卷积的outchannel
    # layer: module.encoder.layer1.0.downsample.0.weight, channel: 256
    # prune 25 channels 10%, acc: 0.9596169858451291
    # layer: module.encoder.layer2.0.downsample.0.weight, channel: 512
    # prune 153 channels 30%, acc: 0.9567027477102414
    # layer: module.encoder.layer3.0.downsample.0.weight, channel: 1024
    # prune 614 channels 60%, acc: 0.9512905911740216
    # layer: module.encoder.layer4.0.downsample.0.weight, channel: 2048
    # prune 1638 channels 80%, acc: 0.9655495420482931
    resnet_prune_layers = [ 'module.encoder.layer1.0.downsample.0.weight', 
    'module.encoder.layer3.0.downsample.0.weight', 
    'module.encoder.layer4.0.downsample.0.weight'
    ] 
    resnet_prune_channels = [int(256*0.3), 
    int(1024*0.7), 
    int(2048*0.9)]

    net = get_net(config)

    weight, net = simple_prune_final(config, net, simple_prune_layers, simple_prune_channels, simple_prune_affected_layers)
    weight, net = group_prune_final(config, weight, net, group_prune_layers, group_prune_channels, resnet_prune_layers)
    weight, net = resnet_prune_final(config, weight, net, resnet_prune_layers, resnet_prune_channels)
    net = torch.nn.DataParallel(net)
    net = net.cuda()
    net.load_state_dict(weight)
    config.cycle_num = 5
    config.cycle_inter = 10

    retrain(config, net)


def main(config):
    if config.mode == 'infer_test':
        config.pretrained_model = r'global_min_acer_model.pth'
        final_prune(config)

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
