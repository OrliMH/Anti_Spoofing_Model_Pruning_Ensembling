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
`python train_CyclicLR.py ` 
fusion mode:
`python train_Fusion_CyclicLR.py `  

### model pruning
Currently only single mode model pruning is supported.  
1. To get results of pruning different conv layers with different pruning rates:  
`python prune_filters.py `  
2. Set the prune rates and the layers to prune, and then prune the model:  
`python final_prune.py `  
3. Retrain the pruned model:  
`python re_train_model_A_color_32.py `
4. Compare metrics and FLOPS:  
`python metric_compare.py `
`python compare_flops.py `
Here is the result of metrics before and after pruning for model A color 32:
+------------+----------+----------+  
| Metric     | Before   | After    |  
+------------+----------+----------+  
| loss       | 0.136110 | 0.216070 |  
| acer       | 0.033845 | 0.053242 |  
| acc        | 0.965862 | 0.933493 |  
| correct    | 0.966283 | 0.934313 |  
| Parameters | 17409084 | 3031974  |  
| Inference  | 0.0280   | 0.0141   |  
+------------+----------+----------+  








