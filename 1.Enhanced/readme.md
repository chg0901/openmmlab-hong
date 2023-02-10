# Training weight file
[epoch_200.pth(google drive)](https://drive.google.com/file/d/1hKHJL4h-KFrjmKVZ-Uzh5mE-UN8eoPLu/view?usp=sharing)


------------------------------------------
# Environment
```
mmcls    0.0.0rc5
mmengine 0.5.0
mmcv     2.0.0rc3
```

------------------------------------------
# Model: resnet18
`load_from ='./resnet18_batch256_imagenet_20200708-34ab8f90.pth' `

## Training commend
```
python tools/train.py configs/resnet/resnet18_cifar_hong.py \
       --work-dir work_dirs/cifar10
```

------------------------------------------
# Test result
```
Loads checkpoint by local backend from path: work_dirs/cifar10/epoch_199.pth
02/06 11:57:11 - mmengine - INFO - Load checkpoint from work_dirs/cifar10/epoch_199.pth
02/06 11:57:41 - mmengine - INFO - Epoch(test) [100/625]    eta: 0:02:37  time: 0.2982  data_time: 0.2702  memory: 1201  
02/06 11:58:11 - mmengine - INFO - Epoch(test) [200/625]    eta: 0:02:08  time: 0.3120  data_time: 0.2836  memory: 1201  
02/06 11:58:42 - mmengine - INFO - Epoch(test) [300/625]    eta: 0:01:38  time: 0.2993  data_time: 0.2711  memory: 1201  
02/06 11:59:12 - mmengine - INFO - Epoch(test) [400/625]    eta: 0:01:08  time: 0.2967  data_time: 0.2685  memory: 1201  
02/06 11:59:42 - mmengine - INFO - Epoch(test) [500/625]    eta: 0:00:37  time: 0.3011  data_time: 0.2729  memory: 1201  
02/06 12:00:12 - mmengine - INFO - Epoch(test) [600/625]    eta: 0:00:07  time: 0.2948  data_time: 0.2667  memory: 1201  
02/06 12:00:19 - mmengine - INFO - Epoch(test) [625/625]  accuracy/top1: 95.0600 
```


## Test commend
```
python tools/test.py configs/resnet/resnet18_cifar_hong.py  \
        work_dirs/cifar10/epoch_199.pth 
```

# Zhihu Link

[OpenMMLab AI 实战营 - 知乎](https://www.zhihu.com/column/c_1605019904180232192)

[MMClassification Enhanced Assignment （Cifar10 & mmcls1.x） - 知乎](https://zhuanlan.zhihu.com/p/603633490)



