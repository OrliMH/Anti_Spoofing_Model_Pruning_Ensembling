# Anti-Spoofing Model Pruning and Ensembling
This repo is based on https://github.com/SeuTao/CVPR19-Face-Anti-spoofing  
I add the pruning part for color mode model.  
The pruning part code originates from my gitee repo https://gitee.com/OrliMH/living-body-detection-and-face-recognition-under-occlusion    


### Dataset  
name:CASIA-SURF    
intro:https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_A_Dataset_and_Benchmark_for_Large-Scale_Multi-Modal_Face_Anti-Spoofing_CVPR_2019_paper.pdf  
The dataset link is supported in the above intro paper.

### Alter data path  
After downloading the dataset, data path should be altered to the location you data downloaded to.  
Data paths variables are in the file process/data_helper.py.

### Model training  
sigle mode:  
`python train_CyclicLR.py `   
fusion mode:  
`python train_Fusion_CyclicLR.py `  
&emsp; &ensp;&ensp;&ensp;&ensp;&emsp; &ensp;&ensp;&ensp;&ensp;&emsp; &ensp;&ensp;&ensp;&ensp;&emsp; &ensp;&ensp;&ensp;acer  
baseline_color_32    &emsp;       0.0491917713059422  
Model_A_color_32     &emsp;       0.033844829059388806  
baseline_color_64    &emsp;       0.1065163791952416  
model_A_color_64     &emsp;       0.06830342471052375  
baseline_no_SE_fusion_32 &emsp;   0.0003779860901118839  
baseline_fusion_32     &emsp;     0.00007559721802237678(7.559721802237678e-05)  
model_A_fusion_32      &emsp;     0.0006047777441790142  

### Model pruning
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

Below is the result of metrics before and after pruning for model A color 32:  
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
Bellow is the metrics and FLOPs before and after pruning for model A color 64 with different pruning rates and layers comparing model A color 32:  
+------------+----------+----------+  
| Metric     | Before   | After    |  
+------------+----------+----------+  
| loss       | 0.175365 | 0.207033 |  
| acer       | 0.040642 | 0.061712 |  
| acc        | 0.949833 | 0.927873 |  
| correct    | 0.950452 | 0.928762 |  
| Parameters | 17409084 | 3169363  |  
| Inference  | 0.0106   | 0.0115   |  
+------------+----------+----------+  
FLOPS:  
before:  207,262,624 FLOPs or approx. 0.21 GFLOPs  
after:    105,704,104 FLOPs or approx. 0.11 GFLOPs   

### Ensemble
`python ensemble.py`  
&emsp; &ensp;&ensp;&ensp;&ensp;loss     &emsp;       acer     &emsp;      acc     &emsp;          correct  
ensemble  &ensp;    0.009825291112065315&ensp; 0.001360749924402782 &ensp;0.9981265611990008&ensp; 0.9981265611990008  
single   &ensp;     0.007655922789126635&ensp; 0.0011339582703356517 &ensp;0.9984388009991674&ensp; 0.9984388009991674  









