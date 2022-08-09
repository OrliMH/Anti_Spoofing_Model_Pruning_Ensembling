import argparse
import torch
import os
from process.augmentation import *
from process.data import *
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


class ColorDataset(Dataset):
    def __init__(self, mode, modality='color', fold_index=-1, image_size=128, augment = None, augmentor = None, balance = True, image=None):
        super(ColorDataset, self).__init__()
        print('fold: '+str(fold_index))
        print(modality)

        self.mode       = mode
        self.modality = modality

        self.augment = augment
        self.augmentor = augmentor
        self.balance = balance

        self.channels = 3
        self.train_image_path = TRN_IMGS_DIR
        self.test_image_path = TST_IMGS_DIR
        self.image_size = image_size
        self.fold_index = fold_index
        self.image = image

    def getitem(self):

        if self.fold_index is None:
            print('WRONG!!!!!!! fold index is NONE!!!!!!!!!!!!!!!!!')
            return
        
        RESIZE_SIZE = 112
        test_id ="random test id"
        #print("before resize,self.image.shape:{}".format(self.image.shape))
        image = cv2.resize(self.image,(RESIZE_SIZE,RESIZE_SIZE))
        #print("inside getitem, after RESIZE_SIZE, image.shape:{}".format(image.shape))
        image = self.augment(image, target_shape=(self.image_size, self.image_size, 3), is_infer = True)
        #print("after self.image_size, image.shape:{}".format(image.size()))
        n = len(image)
        image = np.concatenate(image,axis=0)
        image = np.transpose(image, (0, 3, 1, 2))
        image = image.astype(np.float32)
        image = image.reshape([n, self.channels, self.image_size, self.image_size])
        image = image / 255.0

        return torch.FloatTensor(image), test_id


    def __len__(self):
        return self.num_data

def run_test(config, image):

    out_dir = './models'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model
    augment = get_augment(config.image_mode)

    ## net ---------------------------------------
    net = get_model(model_name=config.model, num_class=2, is_first_bn=True)
    net = torch.nn.DataParallel(net)
    #net =  net.cuda()

    if initial_checkpoint is not None:
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))


    ## net ---------------------------------------
    #net =  net.cuda()

    test_dataset = ColorDataset(mode = 'test', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index,augment=augment, image=image)


    net.eval()

    print('infer!!!!!!!!!')
   
    valid_num  = 0
    probs = []
    
    #print("len of test_dataset:{}".format(len(test_dataset)))

    input, truth = test_dataset.getitem() # (input, truth)
    #print("inside test_dataset")
    n,c,w,h = input.size()
    print("n,c,w,h:")
    print(n, c, w, h)
    #print("n, c, w, h:")
    #print(n,c,w,h)
    #input = input.view(b*n,c,w,h)
    #input = input.cuda()

    with torch.no_grad():
        logit,_,_   = net(input)
        logit = logit.view(1,n,2) # logit [b, n, 2]
        logit = torch.mean(logit, dim = 1, keepdim = False) # [b, 2]
        prob = F.softmax(logit, 1) # prob 也是 [b, 2]

    valid_num += len(input) # accumulate all the infer samples; each sample contains 3 modal 
    probs.append(prob.data.cpu().numpy()) # [m, b, 2]
    print("outside test_dataset")
    probs = np.concatenate(probs) # [m, b, 2] --> [m*b, 2]
    return probs[:, 1] # [m*b]

    
 
