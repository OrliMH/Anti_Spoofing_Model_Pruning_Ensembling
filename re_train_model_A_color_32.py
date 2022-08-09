#coding:utf-8
import os
# os.environ['CUDA_VISIBLE_DEVICES'] =  '4,5,6,7' #'3,2,1,0'
import sys
sys.path.append("..")
sys.path.append("./process")
sys.path.append("./model")


import argparse
from process.data import *
from process.augmentation import *
from metric import *
from loss.cyclic_lr import CosineAnnealingLR_with_Restart



# 不同得数据增强，切片方法
def get_augment(image_mode):
    if image_mode == 'color':
        augment = color_augumentor
    elif image_mode == 'depth':
        augment = depth_augumentor
    elif image_mode == 'ir':
        augment = ir_augumentor
    return augment

# 训练
def retrain(config, prune_net):
    # 数据，模型，训练，
    # import pdb
    # pdb.set_trace()
    out_dir = './models'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name) # ./models/model_A_color_64
    initial_checkpoint = config.pretrained_model
    criterion  = softmax_cross_entropy_criterion

    ## setup  -----------------------------------------------------------------------------
    if not os.path.exists(out_dir +'/checkpoint'):
        os.makedirs(out_dir +'/checkpoint')
    if not os.path.exists(out_dir +'/backup'):
        os.makedirs(out_dir +'/backup')
    if not os.path.exists(out_dir +'/backup'):
        os.makedirs(out_dir +'/backup')

    log = Logger()
    log.open(os.path.join(out_dir,config.model_name+'_retrain.txt'),mode='a') # ./models/model_A_color_64/model_A_color_64_retrain.txt
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... xxx baseline  ... \n')
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    augment = get_augment(config.image_mode)
    train_dataset = FDDataset(mode = 'train', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index,augment=augment)
    train_loader  = DataLoader(train_dataset,
                                shuffle=True,
                                batch_size  = config.batch_size,
                                drop_last   = True,
                                num_workers = 4)

    valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index,augment=augment)
    valid_loader  = DataLoader( valid_dataset,
                                shuffle=False,
                                batch_size = config.batch_size // 36,
                                drop_last  = False,
                                num_workers = 4)

    assert(len(train_dataset)>=config.batch_size)
    log.write('batch_size = %d\n'%(config.batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')
    log.write('** prune_net setting **\n')

    



    log.write('%s\n'%(type(prune_net)))
    log.write('criterion=%s\n'%criterion)
    log.write('\n')

    iter_smooth = 20
    start_iter = 0
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('                                  |------------ VALID -------------|-------- TRAIN/BATCH ----------|         \n')
    log.write('model_name   lr   iter  epoch     |     loss      acer      acc    |     loss              acc     |  time   \n')
    log.write('----------------------------------------------------------------------------------------------------\n')

    iter = 0
    i    = 0

    train_loss = np.zeros(6, np.float32)
    valid_loss = np.zeros(6, np.float32)
    batch_loss = np.zeros(6, np.float32)

    start = timer()
    #优化需要优化得模块
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, prune_net.parameters()),
                          lr=0.1, momentum=0.9, weight_decay=0.0005)

    # 学习率调整策略
    sgdr = CosineAnnealingLR_with_Restart(optimizer,
                                          T_max=config.cycle_inter,
                                          T_mult=1,
                                          model=prune_net,
                                          out_dir='../input/',
                                          take_snapshot=False,
                                          eta_min=1e-3)

    global_min_acer = 1.0
    for cycle_index in range(config.cycle_num):
        print('cycle index: ' + str(cycle_index))
        min_acer = 1.0

        for epoch in range(0, config.cycle_inter):
            sgdr.step()
            lr = optimizer.param_groups[0]['lr']
            print('lr : {:.4f}'.format(lr))

            sum_train_loss = np.zeros(6,np.float32)
            sum = 0
            optimizer.zero_grad()

            for input, truth in train_loader:
                iter = i + start_iter

                # one iteration update  -------------
                prune_net.train()
                #input = input.cuda()
                #truth = truth.cuda()
                # import pdb
                # pdb.set_trace()
                # print("input.shape={}".format(input.shape))

                logit,_,_ = prune_net.forward(input)
                truth = truth.view(logit.shape[0])
                truth = truth.cuda()

                loss  = criterion(logit, truth)
                precision,_ = metric(logit, truth)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # print statistics  ------------
                batch_loss[:2] = np.array(( loss.item(), precision.item(),))

                sum += 1
                if iter%iter_smooth == 0:
                    train_loss = sum_train_loss/sum
                    sum = 0
                i=i+1

            if epoch >= config.cycle_inter // 2:
                prune_net.eval()
                valid_loss,_ = do_valid_test(prune_net, valid_loader, criterion)
                prune_net.train()

                if valid_loss[1] < global_min_acer and epoch > 0:
                    global_min_acer = valid_loss[1]
                    prune_net_pth_pth = os.path.join(out_dir, "entire_prune_net_checkpoint")
                    ckpt_name = prune_net_pth_pth + 'global_min_acer_model.pth'
                    torch.save(prune_net, ckpt_name)
                    # # 保存整个网络
                    # torch.save(net, PATH) 
                    # # 保存网络中的参数, 速度快，占空间少
                    # torch.save(net.state_dict(),PATH)
                    # #--------------------------------------------------
                    # #针对上面一般的保存方法，加载的方法分别是：
                    # model_dict=torch.load(PATH)
                    # model_dict=model.load_state_dict(torch.load(PATH))
                    log.write('save global min acer model: ' + str(min_acer) + '\n')

            asterisk = ' '
            log.write(config.model_name+' Cycle %d: %0.4f %5.1f %6.1f | %0.6f  %0.6f  %0.3f %s  | %0.6f  %0.6f |%s \n' % (
                cycle_index, lr, iter, epoch,
                valid_loss[0], valid_loss[1], valid_loss[2], asterisk,
                batch_loss[0], batch_loss[1],
                time_to_str((timer() - start), 'min')))
            # batch_loss[0]是loss
            # batch_loss[1]是准确率

    log.write('end training! \n')