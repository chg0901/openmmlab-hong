# Training weight file
[latest.pth(google drive)](https://drive.google.com/file/d/13oa80uTYgj0RfjkEqqBkr9iPE-_H8kDe/view?usp=sharing)

# [Project JupyterNotebook with nbviewer](https://nbviewer.org/github/chg0901/openmmlab-hong/blob/main/2.Basic/balloon_hong.ipynb)
[JupyterNotebook balloon_hong.ipynb](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic/balloon_hong.ipynb)

# Results

|   bbox_mAP  | bbox_mAP_50 | bbox_mAP_75 |  bbox_mAP_s |  bbox_mAP_m |  bbox_mAP_l |
| :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
|    0.6838   |   0.7841    |    0.7477   |    0.0000   |    0.3219   |    0.8186   |
    
|  segm_mAP   | segm_mAP_50 | segm_mAP_75 |  segm_mAP_s |  segm_mAP_m |  segm_mAP_l |
| :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
|   0.6678    |   0.7603    |    0.7137   |    0.0000   |    0.2712   |    0.8007   |
    
    
```
bbox_mAP_copypaste: 0.6838 0.7841 0.7477 0.0000 0.3219 0.8186, 
segm_mAP_copypaste: 0.6678 0.7603 0.7137 0.0000 0.2712 0.8007
```

# Results with screenshots and Gifs of the test video
![Unprocessed screenshot ](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic/before.png)
![processed screenshot ](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic/after.png)
![processed Gif 1](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic/result%5B00_00_01--00_00_06%5D.gif)
![processed Gif 2](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic/result%5B00_00_00--00_00_06%5D2.gif)

# Results with test images
![picture1](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic/work_dirs/mask_rcnn_r50_fpn_2x_coco_balloon/show/16335852991_f55de7958d_k.jpg)
![picture2](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic/work_dirs/mask_rcnn_r50_fpn_2x_coco_balloon/show/24631331976_defa3bb61f_k.jpg)
![picture3](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic/work_dirs/mask_rcnn_r50_fpn_2x_coco_balloon/show/3825919971_93fb1ec581_b.jpg)
![picture4](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic/work_dirs/mask_rcnn_r50_fpn_2x_coco_balloon/show/410488422_5f8991f26e_b.jpg)


------------------------------------------
# Environment
```
TorchVision: 0.7.0
OpenCV: 4.7.0
MMCV: 1.7.1
MMCV Compiler: GCC 7.3
MMCV CUDA Compiler: 10.2
MMDetection: 2.28.0+1b7d778

Python: 3.8.16 (default, Jan 17 2023, 23:13:24) [GCC 11.2.0]
CUDA available: True
GPU 0: NVIDIA GeForce GTX 1080 Ti
CUDA_HOME: /home/cine/miniconda3/envs/mmlab1
NVCC: Cuda compilation tools, release 11.6, V11.6.124
GCC: gcc (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0
PyTorch: 1.6.0
```


## Other details are shown in the jupyter notebook

[训练模型的配置文件 mask_rcnn_r50_fpn_2x_coco_balloon.py ](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic/mask_rcnn_r50_fpn_2x_coco_balloon.py)

[训练好的模型文件，latest.pth, google drive](https://drive.google.com/file/d/13oa80uTYgj0RfjkEqqBkr9iPE-_H8kDe/view?usp=sharing)

[特效制作后的视频文件  result.mp4](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic/result.mp4)

[log文件夹](https://github.com/chg0901/openmmlab-hong/tree/main/2.Basic/work_dirs/mask_rcnn_r50_fpn_2x_coco_balloon)

[JupyterNotebook balloon_hong.ipynb](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic/balloon_hong.ipynb)

[CoCo数据集制作python文件 balloon2CoCoFormat.py](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic/balloon2CoCoFormat.py)

[生成的COCO数据集json文件： train.json](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic/train.json)   [val.json](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic/val.json)

[测试集预测图片文件夹](https://github.com/chg0901/openmmlab-hong/tree/main/2.Basic/work_dirs/mask_rcnn_r50_fpn_2x_coco_balloon/show)

[测试视频制作python文件 video.py](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic/video.py)


# Zhihu Link


[OpenMMLab 实战营打卡 - 【第 4 课】目标检测与MMDetection - 知乎](https://zhuanlan.zhihu.com/p/603989451)

[OpenMMLab 实战营打卡 - 【第 5 课】MMDetection3.x - 知乎](https://zhuanlan.zhihu.com/p/604488260?)

[OpenMMLab AI 实战营 - 知乎](https://www.zhihu.com/column/c_1605019904180232192)




