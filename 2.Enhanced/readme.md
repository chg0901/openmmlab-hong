# MMDetection Enhanced Assignment
## Training and testing files

[训练模型的配置文件 job4_cascade_rcnn_x101_32x4d_fpn_1x_coco.py ](https://github.com/chg0901/openmmlab-hong/blob/main/2.Enhanced/job4_cascade_rcnn_x101_32x4d_fpn_1x_coco.py)

[训练好的模型文件 epoch_1.pth, google drive](https://drive.google.com/file/d/1TdJb04quMPYwJvItAin4ETKroWBXaGku/view?usp=share_link)

[训练日志文件 None.log.json](https://github.com/chg0901/openmmlab-hong/blob/main/2.Enhanced/None.log.json)


## Results with videos and screenshots
![Unprocessed video ](https://github.com/chg0901/openmmlab-hong/blob/main/2.Enhanced/57906_000718_Endzone.mp4)

![processed video](https://github.com/chg0901/openmmlab-hong/blob/main/2.Enhanced/57906_000718_Endzone_fps60.mp4)

![Unprocessed screenshoot](https://github.com/chg0901/openmmlab-hong/blob/main/2.Enhanced/%E6%88%AA%E5%9B%BE%202023-02-11%2005-06-48.png)

![processed screenshoot](https://github.com/chg0901/openmmlab-hong/blob/main/2.Enhanced/%E6%88%AA%E5%9B%BE%202023-02-11%2005-07-37.png)


## [Project JupyterNotebook with nbviewer](https://nbviewer.org/github/chg0901/openmmlab-hong/blob/main/2.Enhanced/mmdet-cascadercnn-helmet-detection-for-beginners.ipynb)
[mmdet-cascadercnn-helmet-detection-for-beginners.ipynb](https://github.com/chg0901/openmmlab-hong/blob/main/2.Enhanced/mmdet-cascadercnn-helmet-detection-for-beginners.ipynb)

This [notebook](https://www.kaggle.com/code/chg0901/mmdet-cascadercnn-helmet-detection-for-beginners/edit) is based on the notebook in this [link](https://www.kaggle.com/code/eneszvo/mmdet-cascadercnn-helmet-detection-for-beginners/notebook)

##### [Data and Task](https://www.kaggle.com/competitions/nfl-health-and-safety-helmet-assignment%20%20%20)
`In this competition, you are tasked with assigning the correct player from game footage. Each play has two associated videos, showing a sideline and endzone view, and the videos are aligned so that frames correspond between the videos. `
##### Model: CascadeRCNN

I use this notebook to learn how to use jupyter notebook to modify the model config file.

This notebook also show me ways to edit the video files.

I take the original model and config, but I made several modification when I run it as following:

1.  set the device in config for GPU
```
# AttributeError: 'ConfigDict' object has no attribute 'device'
# https://github.com/open-mmlab/mmdetection/issues/7901#issuecomment-1194858228

def is_mlu_available():
    """Returns a bool indicating if MLU is currently available."""
    return hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()


def get_device():
    """Returns an available device, cpu, cuda or mlu."""
    is_device_available = {
        'cuda': torch.cuda.is_available(),
        'mlu': is_mlu_available()
    }
    device_list = [k for k, v in is_device_available.items() if v]
    return device_list[0] if len(device_list) == 1 else 'cpu'

cfg.device = get_device()

```
2. modify the frame prediction codes, we won't load the model in the loop

from 
```
for f in tqdm(os.listdir(frame_dir)):
    img = f'{frame_dir}/{f}'
    # the model is initialized and deleted each time because of RAM usage
    model = init_detector(cfg, checkpoint, device='cuda:0')
    # get results
    result = inference_detector(model, img)
    # save image with bboxes into out_file
    model.show_result(img, result, out_file=os.path.join(frame_bbox_dir,f))
    del result, model
    gc.collect()
```
to
```
# the model is initialized and deleted each time because of RAM usage
model = init_detector(cfg, checkpoint, device='cuda:0')

for f in tqdm(os.listdir(frame_dir)):
    img = f'{frame_dir}/{f}'
    # get results
    result = inference_detector(model, img)
    # save image with bboxes into out_file
    model.show_result(img, result, out_file=os.path.join(frame_bbox_dir,f))
    del result
    gc.collect()
del result, model
gc.collect()
```


## Results
I just train the model for 1 epoch

|   bbox_mAP  | bbox_mAP_50 | bbox_mAP_75 |  bbox_mAP_s |  bbox_mAP_m |  bbox_mAP_l |
| :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
|    0.0592   |   -1.0000   |   -1.0000   |    0.0576   |    0.1108   |   -1.0000   |


    
```
# 2023-02-10 17:10:20,075 - mmdet - INFO - Epoch [1][2380/2485]	lr: 1.131e-05, eta: 0:04:20, time: 2.484, data_time: 0.016, memory: 12826, loss_rpn_cls: 0.0151, loss_rpn_bbox: 0.0137, s0.loss_cls: 0.1825, s0.acc: 93.7842, s0.loss_bbox: 0.0879, s1.loss_cls: 0.1009, s1.acc: 93.2470, s1.loss_bbox: 0.1275, s2.loss_cls: 0.0569, s2.acc: 91.8304, s2.loss_bbox: 0.0959, loss: 0.6805
# 2023-02-10 17:11:09,822 - mmdet - INFO - Epoch [1][2400/2485]	lr: 7.480e-06, eta: 0:03:31, time: 2.487, data_time: 0.016, memory: 12826, loss_rpn_cls: 0.0172, loss_rpn_bbox: 0.0156, s0.loss_cls: 0.1926, s0.acc: 93.2568, s0.loss_bbox: 0.0909, s1.loss_cls: 0.1081, s1.acc: 92.5061, s1.loss_bbox: 0.1341, s2.loss_cls: 0.0600, s2.acc: 91.3712, s2.loss_bbox: 0.0956, loss: 0.7142
# 2023-02-10 17:11:59,461 - mmdet - INFO - Epoch [1][2420/2485]	lr: 4.449e-06, eta: 0:02:41, time: 2.482, data_time: 0.016, memory: 12826, loss_rpn_cls: 0.0194, loss_rpn_bbox: 0.0146, s0.loss_cls: 0.1880, s0.acc: 93.7183, s0.loss_bbox: 0.0920, s1.loss_cls: 0.1055, s1.acc: 92.9364, s1.loss_bbox: 0.1376, s2.loss_cls: 0.0589, s2.acc: 91.7401, s2.loss_bbox: 0.0972, loss: 0.7132
# 2023-02-10 17:12:49,309 - mmdet - INFO - Epoch [1][2440/2485]	lr: 2.213e-06, eta: 0:01:51, time: 2.492, data_time: 0.017, memory: 12826, loss_rpn_cls: 0.0134, loss_rpn_bbox: 0.0121, s0.loss_cls: 0.1830, s0.acc: 93.7915, s0.loss_bbox: 0.0913, s1.loss_cls: 0.1019, s1.acc: 93.2016, s1.loss_bbox: 0.1377, s2.loss_cls: 0.0578, s2.acc: 91.8937, s2.loss_bbox: 0.0973, loss: 0.6945
# 2023-02-10 17:13:38,990 - mmdet - INFO - Epoch [1][2460/2485]	lr: 7.752e-07, eta: 0:01:02, time: 2.484, data_time: 0.016, memory: 12826, loss_rpn_cls: 0.0211, loss_rpn_bbox: 0.0159, s0.loss_cls: 0.2106, s0.acc: 92.6221, s0.loss_bbox: 0.0960, s1.loss_cls: 0.1194, s1.acc: 91.8665, s1.loss_bbox: 0.1430, s2.loss_cls: 0.0656, s2.acc: 90.7499, s2.loss_bbox: 0.1017, loss: 0.7733
# 2023-02-10 17:14:28,627 - mmdet - INFO - Epoch [1][2480/2485]	lr: 1.360e-07, eta: 0:00:12, time: 2.482, data_time: 0.017, memory: 12826, loss_rpn_cls: 0.0159, loss_rpn_bbox: 0.0168, s0.loss_cls: 0.2023, s0.acc: 93.0518, s0.loss_bbox: 0.0957, s1.loss_cls: 0.1162, s1.acc: 91.9426, s1.loss_bbox: 0.1381, s2.loss_cls: 0.0624, s2.acc: 91.1122, s2.loss_bbox: 0.0906, loss: 0.7379
# 2023-02-10 17:14:41,050 - mmdet - INFO - Saving checkpoint at 1 epochs

# [>>>>>>>>>>>>>>>>>>>>>>>>>>] 9947/9947, 5.3 task/s, elapsed: 1873s, ETA:     0s

# 2023-02-10 17:46:07,009 - mmdet - INFO - Evaluating bbox...

# Loading and preparing results...
# DONE (t=1.82s)
# creating index...
# index created!
# Running per image evaluation...
# Evaluate annotation type *bbox*
# DONE (t=14.97s).
# Accumulating evaluation results...

# 2023-02-10 17:46:30,642 - mmdet - INFO - 
#  Average Precision  (AP) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ] = 0.059
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = -1.000
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = -1.000
#  Average Precision  (AP) @[ IoU=0.50:0.50 | area= small | maxDets=1000 ] = 0.058
#  Average Precision  (AP) @[ IoU=0.50:0.50 | area=medium | maxDets=1000 ] = 0.111
#  Average Precision  (AP) @[ IoU=0.50:0.50 | area= large | maxDets=1000 ] = -1.000
#  Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ] = 0.462
#  Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=300 ] = 0.462
#  Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=1000 ] = 0.462
#  Average Recall     (AR) @[ IoU=0.50:0.50 | area= small | maxDets=1000 ] = 0.457
#  Average Recall     (AR) @[ IoU=0.50:0.50 | area=medium | maxDets=1000 ] = 0.608
#  Average Recall     (AR) @[ IoU=0.50:0.50 | area= large | maxDets=1000 ] = -1.000

# DONE (t=3.93s).

# 2023-02-10 17:46:31,214 - mmdet - INFO - Epoch(val) [1][9947]	
# bbox_mAP: 0.0592, bbox_mAP_50: -1.0000, bbox_mAP_75: -1.0000, 
# bbox_mAP_s: 0.0576, bbox_mAP_m: 0.1108, bbox_mAP_l: -1.0000, 
# bbox_mAP_copypaste: 0.0592 -1.0000 -1.0000 0.0576 0.1108 -1.0000

bbox_mAP_copypaste: 0.0592 -1.0000 -1.0000 0.0576 0.1108 -1.0000
```


## Zhihu Link


[OpenMMLab 实战营打卡 - 【第 4 课】目标检测与MMDetection - 知乎](https://zhuanlan.zhihu.com/p/603989451)

[OpenMMLab 实战营打卡 - 【第 5 课】MMDetection3.x - 知乎](https://zhuanlan.zhihu.com/p/604488260?)

[OpenMMLab AI 实战营 - 知乎](https://www.zhihu.com/column/c_1605019904180232192)


