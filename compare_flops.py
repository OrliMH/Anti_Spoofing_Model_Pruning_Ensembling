import torch
from torchvision.models import resnet18
from prune import *
from prune_filters import *
from prune_valid import *


from pthflops import count_ops

def compute_flops(model):
    # Create a network and a corresponding input

    inp = torch.rand(1, 3, 64, 64).cuda()

    # Count the number of FLOPs
    num = count_ops(model, inp)

    return num
def run_compute(config):

    origin_model=get_net_weight(config)
    origin_model = torch.nn.DataParallel(origin_model)
    origin_model = origin_model.cuda()

    
    out_dir = './models'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name) # ./models/model_A_color_64
    prune_net_pth_pth = os.path.join(out_dir, "entire_prune_net_checkpoint")
    ckpt_name = prune_net_pth_pth + 'global_min_acer_model.pth'
    
    # Model class must be defined somewhere
    prune_model = torch.load(ckpt_name) # model is already parallel 'case checkpoint is parallel and checkpoint contains net structure
    prune_model = prune_model.cuda()
 
    origin_flops = compute_flops(origin_model) # 207,262,624 FLOPs or approx. 0.21 GFLOPs
    prune_flops = compute_flops(prune_model) # 105,704,104 FLOPs or approx. 0.11 GFLOPs

    log = Logger()
    log.open('./flops_before_after_prune.txt',mode='a')
    log.write("\n")
    log.write("origin_flops:")
    log.write("\n")
    log.write(str(origin_flops))
    log.write("\n")
    log.write("prune_flops:")
    log.write(str(prune_flops))




def main(config):
    if config.mode == 'infer_test':
        config.pretrained_model = r'global_min_acer_model.pth'
        run_compute(config)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default = -1)

    parser.add_argument('--model', type=str, default='model_A')
    parser.add_argument('--image_mode', type=str, default='color')
    parser.add_argument('--image_size', type=int, default=64)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cycle_num', type=int, default=10)
    parser.add_argument('--cycle_inter', type=int, default=50)

    parser.add_argument('--mode', type=str, default='infer_test', choices=['train','infer_test'])
    parser.add_argument('--pretrained_model', type=str, default=None)

    config = parser.parse_args()

    print(config)
    main(config)