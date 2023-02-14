# MMSegmentation Basic Assignment

## Training weight file
[latest.pth(google drive)](https://drive.google.com/file/d/1JeUcc66zQ5MvnVNnv68kc5XK03d-IrQF/view?usp=sharing)

## [Project JupyterNotebook with nbviewer](https://nbviewer.org/github/chg0901/openmmlab-hong/blob/main3.Basic/Basic3.ipynb)
[JupyterNotebook Basic3.ipynb](https://github.com/chg0901/openmmlab-hong/blob/main/3.Basic/Basic3.ipynb)

## Results

### 子豪兄test结果
```
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
| background | 99.27 | 99.73 |
| glomeruili | 66.68 | 76.13 |
+------------+-------+-------+
```
aAcc: 99.2800  mIoU: 82.9700  mAcc: 87.9300


### 我的test结果
```
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
| background |  96.4 | 96.66 |
| glomeruili | 29.77 | 84.95 |
+------------+-------+-------+
aAcc: 96.4500  mIoU: 63.0900  mAcc: 90.8100

```
### 我的training结果
```
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
| background | 99.45 | 99.96 |
| glomeruili | 69.83 | 71.21 |
+------------+-------+-------+
aAcc: 99.4600  mIoU: 84.6400  mAcc: 85.5900
test speed
```

------------------------------------------


### Other details are shown in the jupyter notebook

[训练模型的配置文件 new_cfg_Glomeruli3.py ](https://github.com/chg0901/openmmlab-hong/blob/main/3.Basic/new_cfg_Glomeruli3.py)

[iter_300.pth, google drive](https://drive.google.com/file/d/1JeUcc66zQ5MvnVNnv68kc5XK03d-IrQF/view?usp=sharing)

[log文件夹](https://github.com/chg0901/openmmlab-hong/tree/main/3.Basic//work_dirs/)

[JupyterNotebook Basic3.ipynb](https://github.com/chg0901/openmmlab-hong/blob/main/3.Basic/Basic3.ipynb)



## Zhihu Link

[MMSegmentation基础实验with Kaggle小鼠肾小球切片语义分割 - 知乎](https://zhuanlan.zhihu.com/p/606402314)

[AI 训练营回顾 它山之石可以攻玉 - 知乎](https://zhuanlan.zhihu.com/p/605411327)

[OpenMMLab 实战营打卡 - 【第 7 课】 - 知乎](https://zhuanlan.zhihu.com/p/605254541)

[OpenMMLab 实战营打卡 - 【第 6 课】 - 知乎](https://zhuanlan.zhihu.com/p/604931171)

[OpenMMLab AI 实战营 - 知乎](https://www.zhihu.com/column/c_1605019904180232192)




