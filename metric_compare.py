from prune import *
from prune_filters import *
from prune_valid import *
from terminaltables import AsciiTable


def metric_compare(config):    
    origin_model=get_net_weight(config)
    origin_model = origin_model.cuda()


    
    out_dir = './models'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name) # ./models/model_A_color_64
    prune_net_pth_pth = os.path.join(out_dir, "entire_prune_net_checkpoint")
    ckpt_name = prune_net_pth_pth + 'global_min_acer_model.pth'
    
    # Model class must be defined somewhere
    prune_model = torch.load(ckpt_name) # model is already parallel 'case checkpoint is parallel and checkpoint contains net structure
    prune_model = prune_model.cuda()


    log = Logger()
    log.open('./metric_before_after_prune.txt',mode='a')
    log.write("\n")

    # 对比前向推理时间
    random_input = torch.rand((1, 3, 64, 64)).cuda() # 1, 3, 64, 64

    def obtain_avg_forward_time(input, model, repeat=200):
        model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                output = model(input)
        avg_infer_time = (time.time() - start) / repeat

        return avg_infer_time, output

    print('testing Inference time...')
    origin_forward_time, origin_output = obtain_avg_forward_time(random_input, origin_model)
    prune_forward_time, prune_output = obtain_avg_forward_time(random_input, prune_model)
    print("origin_infer_time:{}".format(origin_forward_time))
    print("prune_infer_time:{}".format(prune_forward_time))
    
    
    # 对比各种指标
    origin_loss, origin_acer, origin_acc, origin_correct = get_metric(config, origin_model)
    loss, acer, acc, correct = get_metric(config, prune_model)
    print("origin metric:")
    print(origin_loss, origin_acer, origin_acc, origin_correct)
    print("prune metric:")
    print(loss, acer, acc, correct)
    
    # 对比参数量
    obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])

    origin_nparameters = obtain_num_parameters(origin_model)
    prune_nparameters = obtain_num_parameters(prune_model)

    # 汇聚成表
    metric_table = [
        ["Metric", "Before", "After"],
        ["loss", f'{origin_loss:.6f}', f'{loss:.6f}'],
        ["acer", f'{origin_acer:.6f}', f'{acer:.6f}'],
        ["acc", f'{origin_acc:.6f}', f'{acc:.6f}'],
        ["correct", f'{origin_correct:.6f}', f'{correct:.6f}'],
        ["Parameters", f"{origin_nparameters}", f"{prune_nparameters}"], # prune/origin=0.17416045554
        ["Inference", f'{origin_forward_time:.4f}', f'{prune_forward_time:.4f}'] # prune/origin=0.50357142857
    ]
    log.write(AsciiTable(metric_table).table)

def main(config):
    if config.mode == 'infer_test':
        config.pretrained_model = r'global_min_acer_model.pth'
        metric_compare(config)

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