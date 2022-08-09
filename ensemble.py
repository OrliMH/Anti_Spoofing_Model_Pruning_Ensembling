import os
# os.environ['CUDA_VISIBLE_DEVICES'] =  '6'
import sys
sys.path.append("..")
sys.path.append("./process")
sys.path.append("./model")
import argparse
from process.data_fusion import *
from process.augmentation import *
from metric import *
from loss.cyclic_lr import CosineAnnealingLR_with_Restart

def get_model(model_name, num_class):
    if model_name == 'baseline':
        from model_fusion.model_baseline_SEFusion import FusionNet
    elif model_name == 'model_A':
        from model_fusion.FaceBagNet_model_A_SEFusion import FusionNet
    elif model_name == 'model_B':
        from model_fusion.FaceBagNet_model_B_SEFusion import FusionNet
    elif model_name == 'resnet18_fusion_baseline':
        from model_fusion.model_baseline_Fusion import FusionNet

    net = FusionNet(num_class=num_class)
    return net

def do_ensamble_valid_test( nets, test_loader, criterion ):
    valid_num  = 0
    losses   = []
    corrects = []
    probs = []
    labels = []
    

    for i, (input, truth) in enumerate(tqdm(test_loader)):
        b,n,c,w,h = input.size()  # n is the num of modal
        input = input.view(b*n,c,w,h) # input: [b*n, c, w, h]

        with torch.no_grad():
            logits = torch.zeros((b,2))
            flag = False
            for net in nets:
                logit,_,_   = net(input)  
                logit = logit.view(b,n,2) # logit [b, n, 2]
                logit = torch.mean(logit, dim = 1, keepdim = False) # logit [b, 2]
                if not flag:
                    logits = logit
                    logits = logits.unsqueeze(dim=0)
                    flag = True
                else:
                    logits = torch.cat((logits, logit.unsqueeze(dim=0)), dim=0)
                # logits: [[b, 2], [b, 2], [b, 2],...] ==> [m, b, 2]

            logit = torch.mean(logits, dim=0) # logits: [m, b, 2]   logit: [b, 2]
            # logits after mean dim=0: logits:[b, 2]
            # mean dim related:
            # if mean dim=0, it saves the dim>0 and does mean operation on corresponding location  
            # import torch
            # a = torch.tensor([[[0.1, 0.9],
            #                 [0.2, 0.8]],
            #                 [[0.3, 0.7],
            #                 [0.4, 0.6]],
            #                 [[0.5, 0.5],
            #                 [0.6, 0.4]]], dtype=torch.float16)
            # print(a.shape)
            # b = torch.mean(a, dim=0)
            # print(b.shape)
            # print(b)
            # print((0.1+0.3+0.5)/3)
            # print((0.9+0.7+0.5)/3)
            # print((0.2+0.4+0.6)/3)
            # print((0.8+0.6+0.4)/3)
            # torch.Size([3, 2, 2])
            # torch.Size([2, 2])
            # tensor([[0.3000, 0.6997],
            #         [0.3997, 0.6001]], dtype=torch.float16)
            # 0.3
            # 0.7000000000000001
            # 0.4000000000000001
            # 0.6


             
            truth = truth.view(logit.shape[0]).cuda() # truth [b]
            loss    = criterion(logit, truth, False) # input: [b, 2] [b] output [b]
            correct, prob = metric(logit, truth) # input: [b, 2] [b] output return correct, prob # acc, 对预测结果做softmax   scaler [N, 2]

        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct).reshape([1]))
        probs.append(prob.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())

    #assert(valid_num == len(test_loader.sampler))
    #----------------------------------------------

    correct = np.concatenate(corrects)
    loss    = np.concatenate(losses)
    loss    = loss.mean()
    correct = np.mean(correct)

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    tpr, fpr, acc = calculate_accuracy(0.5, probs[:,1], labels)
    acer,_,_,_,_ = ACER(0.5, probs[:, 1], labels)

    valid_loss = np.array([
        loss, acer, acc, correct
    ])

    return valid_loss,[probs[:, 1], labels]

def do_single_valid_test(net, test_loader, criterion):
    valid_num  = 0
    losses   = []
    corrects = []
    probs = []
    labels = []


    for i, (input, truth) in enumerate(tqdm(test_loader)):
    # for input, truth in test_loader:
        b,n,c,w,h = input.size()
        input = input.view(b*n,c,w,h)

        #input = input.cuda()
        #truth = truth.cuda()

        with torch.no_grad():
            logit,_,_   = net(input)
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim = 1, keepdim = False)

            truth = truth.view(logit.shape[0]).cuda()
            loss    = criterion(logit, truth, False)
            correct, prob = metric(logit, truth)

        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct).reshape([1]))
        probs.append(prob.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())

    # assert(valid_num == len(test_loader.sampler))
    #----------------------------------------------

    correct = np.concatenate(corrects)
    loss    = np.concatenate(losses)
    loss    = loss.mean()
    correct = np.mean(correct)

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    tpr, fpr, acc = calculate_accuracy(0.5, probs[:,1], labels)
    acer,_,_,_,_ = ACER(0.5, probs[:, 1], labels)

    valid_loss = np.array([
        loss, acer, acc, correct
    ])

    return valid_loss,[probs[:, 1], labels]


def run_ensemble_valid(config):
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = './models'
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoints = config.pretrained_models

    nets = []
    for initial_checkpoint in initial_checkpoints:
        net = get_model(model_name=config.model, num_class=2)
        net = torch.nn.DataParallel(net)
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index)
        valid_loader  = DataLoader( valid_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size,
                                drop_last   = False,
                                num_workers=8)
        criterion = softmax_cross_entropy_criterion
        net.eval()
        nets.append(net)
    valid_loss,out = do_ensamble_valid_test(nets, valid_loader, criterion)
    pth_result_ensemble = os.path.join(out_dir, "ensemble_loss.txt")
    with open(pth_result_ensemble, "w") as f:
        f.write(str(valid_loss[0]) + " " + str(valid_loss[1]) + " " + str(valid_loss[2]) + " " + str(valid_loss[3]))
    f.close()
    

def run_single_valid(config):
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = './models'
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_models[0]

    ## net ---------------------------------------
    net = get_model(model_name=config.model, num_class=2)
    net = torch.nn.DataParallel(net)
    #net =  net.cuda()

    if initial_checkpoint is not None:
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))


    valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index)
    valid_loader  = DataLoader( valid_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size,
                                drop_last   = False,
                                num_workers=8)


    criterion = softmax_cross_entropy_criterion
    net.eval()

    valid_loss,out = do_single_valid_test(net, valid_loader, criterion)
    pth_result_sigle = os.path.join(out_dir,"global_min_acer_model_loss.txt")
    with open(pth_result_sigle, "w") as f:
        f.write(str(valid_loss[0]) + " " + str(valid_loss[1]) + " " + str(valid_loss[2]) + " " + str(valid_loss[3]))
    f.close()
    


def main(config):
    if config.mode == 'ensemble_valid':
        run_ensemble_valid(config)
    else:
        run_single_valid(config)


if __name__ == '__main__':
    pth = ['global_min_acer_model.pth', 'Cycle_0_min_acer_model.pth', 'Cycle_1_min_acer_model.pth', 
    'Cycle_2_min_acer_model.pth', 'Cycle_3_min_acer_model.pth',  'Cycle_4_min_acer_model.pth', 
    'Cycle_5_min_acer_model.pth', 'Cycle_6_min_acer_model.pth', 'Cycle_7_min_acer_model.pth', 
    'Cycle_8_min_acer_model.pth', 'Cycle_9_min_acer_model.pth']
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default = -1)
    parser.add_argument('--model', type=str, default='resnet18_fusion_baseline', choices=['resnet18_fusion_baseline', 'baseline'])
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--image_mode', type=str, default='fusion')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--cycle_num', type=int, default=10)
    parser.add_argument('--cycle_inter', type=int, default=50)

    parser.add_argument('--mode', type=str, default='single_valid', choices=['ensemble_valid', 'single_valid'])
    parser.add_argument('--pretrained_models', type=list, default=pth)

    config = parser.parse_args()
    print(config)
    main(config)