# 新配置继承了基本配置，并做了必要的修改
# _base_ = '../mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py'
_base_ = 'configs/mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py'

# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=80),
        mask_head=dict(num_classes=80)),
    backbone=dict(
        init_cfg=None),
)

# 修改数据集相关配置
data_root = 'data/balloon/'
# metainfo = {
#     'classes': ('balloon', ),
#     'palette': [
#         (220, 20, 60),
#     ]
# }
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        # metainfo=metainfo,
        ann_file='train/train.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        # metainfo=metainfo,
        ann_file='val/val.json',
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'val/val.json')
test_evaluator = val_evaluator

# 使用预训练的 Mask R-CNN 模型权重来做初始化，可以提高模型性能
load_from = 'mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'