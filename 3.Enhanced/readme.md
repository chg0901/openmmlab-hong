# MMSegmentation Enhanced Assignment

## Training weight file
[iter_1200.pth(google drive)](https://drive.google.com/file/d/1rDCJrjuMwckV7nD-tWWKFtUwFhI_Nr3N/view?usp=sharing)

## [Project JupyterNotebook with nbviewer](https://github.com/chg0901/openmmlab-hong/blob/main/3.Enhanced/m2nist.ipynb)
[JupyterNotebook m2nist.ipynb](https://github.com/chg0901/openmmlab-hong/blob/main/3.Enhanced/m2nist.ipynb)

## Data Source and Introduction


**Inspiration**

The M2NIST dataset is released in the hope that it enables users to understand semantic segmentation and perhaps give us more insights about what neural networks learn and in turn leads us to smaller and more robust networks.

**Context**

I created this dataset to teach the basics of fully convolution networks for semantic segmentation of images. 

Most real-world semantic image segmentation tasks require building huge networks that are slow to train and experiment with. 

The dataset was generated by selecting up to 3 random 28px x 28px grayscale images from the MNIST dataset and copying them in to a single 64px(height) x 84px(width) image. 

The digits were pasted so that they did not overlap and no transformations were applied to the original images, so digits in M2NIST maintain the same orientation as the have in MNIST.

**Content**

The dataset has 5000 multi-digit images in combined.npy and 11 segmentation masks for every image in segmented.npy. 

The files can be read in using numpy.load(), for example, as combined=np.load('combined.npy') and segmented = np.load('segmented.npy'). 

The data in combined.npy has shape (5000, 64, 84) while the data in segmented.npy has shape (5000, 64, 84, 11). 

Every element in combined.npy is a grayscale image with up to 3 digits. 

The corresponding element in segmented.npy is a tensor with 64 rows, 84 columns and 11 layers or channels. Each layer or channel is a binary mask. 

The k-th layer (0<=k<9) has 1s wherever the digit k is present in the combined image and 0s everywhere else. 

The last layer k=10 represents background and has 1s wherever there is no digit in the combined image and 0's wherever at pixels where some digit is present in the original image.


Multidigit MNIST(M2NIST) | Kaggle
https://www.kaggle.com/datasets/farhanhubble/multimnistm2nist?datasetId=37151

U-net with softmax/binary_crossentropy | Kaggle
https://www.kaggle.com/code/ryzhovdmitry/u-net-with-softmax-binary-crossentropy

M2NIST Segmentation / U-net | Kaggle
https://www.kaggle.com/code/zhoulingyan0228/m2nist-segmentation-u-net


**一个超小型分割检测数据集 - 知乎**
https://zhuanlan.zhihu.com/p/95518858

**语义分割小数据集  leonardohaig的博客**

1. 利用小型数据集m2nist进行语义分割——(一)数据集介绍  
https://blog.csdn.net/leonardohaig/article/details/10559709
2. 利用小型数据集m2nist进行语义分割——(二)分割网络框架设计
https://blog.csdn.net/leonardohaig/article/details/105597140
3. 利用小型数据集m2nist进行语义分割——(三)代码编写及训练与预测
https://blog.csdn.net/leonardohaig/article/details/105597159

## Results

_此处实验不理想，时间有限，就搜集一些文档资料吧_

1. win10下mmsegmentation的安装训练以及使用mmdeploy部署的全过程_yuanjiaqi_k的博客-CSDN博客

https://blog.csdn.net/yuanjiaqi_k/article/details/126153117


2. 学懂 ONNX，PyTorch 模型部署再也不怕！-阿里云开发者社区

https://developer.aliyun.com/article/914229

3. MMDeploy部署实战系列【第五章】
Windows下Release x64编译mmdeploy(C++)，
对TensorRT模型进行推理 - gy77 - 博客园

https://www.cnblogs.com/gy77/p/16523947.html

4. 记录mmdeploy部署fastscnn到ncnn并量化 - 知乎

https://zhuanlan.zhihu.com/p/567319765

------------------------------------------


### Other details are shown in the jupyter notebook

[训练模型的配置文件 new_cfg2.py ](https://github.com/chg0901/openmmlab-hong/blob/main/3.Enhanced/new_cfg2.py）
[iter_1200.pth(google drive)](https://drive.google.com/file/d/1rDCJrjuMwckV7nD-tWWKFtUwFhI_Nr3N/view?usp=sharing)

[log文件夹](https://github.com/chg0901/openmmlab-hong/tree/main/3.Enhanced/work_dirs/m2nist2）

[JupyterNotebook m2nist.ipynb](https://github.com/chg0901/openmmlab-hong/blob/main/3.Enhanced/m2nist.ipynb）



## Zhihu Link

[MMSegmentation基础实验with Kaggle小鼠肾小球切片语义分割 - 知乎](https://zhuanlan.zhihu.com/p/606402314)

[AI 训练营回顾 它山之石可以攻玉 - 知乎](https://zhuanlan.zhihu.com/p/605411327)

[OpenMMLab 实战营打卡 - 【第 7 课】 - 知乎](https://zhuanlan.zhihu.com/p/605254541)

[OpenMMLab 实战营打卡 - 【第 6 课】 - 知乎](https://zhuanlan.zhihu.com/p/604931171)

[OpenMMLab AI 实战营 - 知乎](https://www.zhihu.com/column/c_1605019904180232192)



