# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1)))

# Modify dataset related settings
data_root = 'data/kitti_tiny/'
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
        # ann_file='train/annotation_coco.json',
        # data_prefix=dict(img='train/')
        ann_file='data/kitti_tiny/train_ann.pkl',
        data_prefix=dict(img="data/kitti_tiny/training/image_2/"),
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        # metainfo=metainfo,
        ann_file='data/kitti_tiny/val_ann.pkl',
        data_prefix=dict(img="data/kitti_tiny/training/image_2/"),
        # ann_file='val/annotation_coco.json',
        # data_prefix=dict(img='val/')
        )
)
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'val/annotation_coco.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
load_from = 'retinanet_r18_fpn_1x_coco_20220407_171055-614fd399.pth' #  None