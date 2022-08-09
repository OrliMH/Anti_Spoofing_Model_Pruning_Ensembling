# Anti-Spoofing Model Pruning and Ensembling
This repo is based on https://github.com/SeuTao/CVPR19-Face-Anti-spoofing  
I add the pruning part for color mode model.  
The pruning part code originates from my gitee repo https://gitee.com/OrliMH/living-body-detection-and-face-recognition-under-occlusion    


### data set  
name:CASIA-SURF    
intro:https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_A_Dataset_and_Benchmark_for_Large-Scale_Multi-Modal_Face_Anti-Spoofing_CVPR_2019_paper.pdf  
The dataset link is supported in the above intro paper.

### alter data path  
After downloading the dataset, data path should be altered to the location you data downloaded to.  
Data paths variables are in the file process/data_helper.py.

### model training  
sigle mode:  
python train_CyclicLR.py  
fusion mode:
python train_Fusion_CyclicLR.py  





