# MMDetection Basic Assignment in MMDetection3.x

## Training weight file
[epoch_24_mmdet3.x.pth (google drive)](https://drive.google.com/file/d/13PvmX7THJF2JA3iad3q0AJma0uYgCzqJ/view?usp=sharing)

## [Project JupyterNotebook with nbviewer](https://nbviewer.org/github/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/balloon_hong_mmdet3M3.ipynb)
[JupyterNotebook balloon_hong_mmdet3M.ipynb](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/balloon_hong_mmdet3M3.ipynb)



### Other details are shown in the jupyter notebook

[训练模型的配置文件 rcnn_r50_fpn_2x_coco_balloon3_new.py ](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/mask-rcnn_r50_fpn_2x_coco_balloon3_new.py)

[测试预训练模型的配置文件 mask-rcnn_r50_fpn_2x_cocoM.py](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/mask-rcnn_r50_fpn_2x_cocoM.py)

[训练好的模型文件，latest.pth, google drive](https://drive.google.com/file/d/13oa80uTYgj0RfjkEqqBkr9iPE-_H8kDe/view?usp=sharing)

[特效制作后的视频文件  result.mp4](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/result.mp4)

[log ](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/work_dirs/20230216_061558/20230216_061558.log)

[CoCo数据集制作python文件 balloon2CoCoFormat3.py](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/balloon2CoCoFormat3.py)

[测试集预测图片文件夹](https://github.com/chg0901/openmmlab-hong/tree/main/2.Basic_mmdet3.x_V2/work_dirs/20230216_061558/show)

[测试视频制作python文件 video3.py](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/video3.py)


------------------------------------------
## Environment
```
    GPU 0: NVIDIA GeForce GTX 1080 Ti
    CUDA_HOME: /home/cine/miniconda3/envs/mmlab2
    NVCC: Cuda compilation tools, release 11.6, V11.6.124
    GCC: gcc (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0
    PyTorch: 1.6.0
    PyTorch compiling details: PyTorch built with:
  - GCC 7.3
  - C++ Version: 201402
  - Intel(R) oneAPI Math Kernel Library Version 2021.4-Product Build 20210904 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v1.5.0 (Git Hash e2ac1fac44c5078ca927cb9b90e1b3066a0b2ed0)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 10.2
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_37,code=compute_37
  - CuDNN 7.6.5
  - Magma 2.5.2
  - Build settings: BLAS=MKL, BUILD_TYPE=Release, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DUSE_VULKAN_WRAPPER -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, USE_CUDA=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, USE_STATIC_DISPATCH=OFF, 

    TorchVision: 0.7.0
    OpenCV: 4.7.0
    MMEngine: 0.5.0
```

## Results  **[Best results]**
```
coco/bbox_mAP: 0.7660  coco/bbox_mAP_50: 0.8820  coco/bbox_mAP_75: 0.8820  
coco/bbox_mAP_s: 0.3530  coco/bbox_mAP_m: 0.6860  coco/bbox_mAP_l: 0.8040  

coco/segm_mAP: 0.7800  coco/segm_mAP_50: 0.8620  coco/segm_mAP_75: 0.8620  
coco/segm_mAP_s: 0.4540  coco/segm_mAP_m: 0.6330  coco/segm_mAP_l: 0.8310
```

^^
||

|   bbox_mAP  | bbox_mAP_50 | bbox_mAP_75 |  bbox_mAP_s |  bbox_mAP_m |  bbox_mAP_l |
| :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
|    0.6838   |   0.7841    |    0.7477   |    0.0000   |    0.3219   |    0.8186   |
    
|  segm_mAP   | segm_mAP_50 | segm_mAP_75 |  segm_mAP_s |  segm_mAP_m |  segm_mAP_l |
| :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
|   0.6678    |   0.7603    |    0.7137   |    0.0000   |    0.2712   |    0.8007   |

```
bbox_mAP_copypaste: 0.766 0.882 0.882 0.353 0.686 0.804
segm_mAP_copypaste: 0.780 0.862 0.862 0.454 0.633 0.831
^^
||
bbox_mAP_copypaste: 0.6838 0.7841 0.7477 0.0000 0.3219 0.8186, 
segm_mAP_copypaste: 0.6678 0.7603 0.7137 0.0000 0.2712 0.8007
```

## Results with screenshots and Gifs of the test video
![Unprocessed screenshot ](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/before.png)
![processed screenshot ](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/after.png)
![processed Gif 1](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/result%5B00_00_01--00_00_06%5D.gif)
![processed Gif 2](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/result%5B00_00_00--00_00_06%5D2.gif)


## Results with test images
![picture1](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/%E6%88%AA%E5%9B%BE%202023-02-12%2018-47-01.png)
![picture2](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/%E6%88%AA%E5%9B%BE%202023-02-12%2018-45-53.png)

![picture1](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/work_dirs/20230216_061558/show/16335852991_f55de7958d_k.jpg)

![picture2](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/work_dirs/20230216_061558/show/24631331976_defa3bb61f_k.jpg)

![picture3](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/work_dirs/20230216_061558/show/3825919971_93fb1ec581_b.jpg)

![picture4](https://github.com/chg0901/openmmlab-hong/blob/main/2.Basic_mmdet3.x_V2/work_dirs/20230216_061558/show/16335852991_f55de7958d_k.jpg)
        

## Zhihu Link

**[使用MMDetection3.x 在Balloon气球数据集上训练并对视频进行实例分割和制作成Color Splash效果 - 知乎](https://zhuanlan.zhihu.com/p/606610273)**

[OpenMMLab AI 实战营 - 知乎](https://www.zhihu.com/column/c_1605019904180232192)

[OpenMMLab 实战营打卡 - 【第 4 课】目标检测与MMDetection - 知乎](https://zhuanlan.zhihu.com/p/603989451)

[OpenMMLab 实战营打卡 - 【第 5 课】MMDetection3.x - 知乎](https://zhuanlan.zhihu.com/p/604488260?)


[OpenMMLab 实战营打卡 - 【第 6 课】语义分割 - 知乎](https://zhuanlan.zhihu.com/p/604931171)

[OpenMMLab 实战营打卡 - 【第 7 课】MMSegmentation - 知乎](https://zhuanlan.zhihu.com/p/605254541)

[MMSegmentation语义分割基础实验 with Kaggle小鼠肾小球Glomeruli-datase数据 - 知乎](https://zhuanlan.zhihu.com/p/606402314)

[AI 训练营回顾 它山之石可以攻玉 - 知乎](https://zhuanlan.zhihu.com/p/605411327)

[Git 命令大全 - 知乎](https://zhuanlan.zhihu.com/p/603981518?)




