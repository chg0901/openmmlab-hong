_base_=['mask_rcnn_r50_fpn_2x_coco.py']

# _base_ = [
#     '../_base_/models/mask_rcnn_r50_fpn.py',
#     '../_base_/datasets/coco_instance.py',
#     '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
# ]


model = dict(roi_head=dict(bbox_head=dict(num_classes=1),mask_head=dict(num_classes=1)))

data = dict(
    train=dict(
        ann_file='data/balloon/train/train.json',
        img_prefix='data/balloon/train/',
        classes=('balloon',)
    ),
    val=dict(
        ann_file='data/balloon/val/val.json',
        img_prefix='data/balloon/val/',
        classes=('balloon',)
    ),
    test=dict(
        ann_file='data/balloon/val/val.json',
        img_prefix='data/balloon/val/',
        classes=('balloon',)
    )
)

# UserWarning: "ImageToTensor" pipeline is replaced by "DefaultFormatBundle" for batch inference.
#     It is recommended to manually replace it in the test data pipeline in your config file.
test_pipeline = [
    dict(transforms=[
            # dict(type='DefaultFormatBundle', keys=['img']),
            dict(type='DefaultFormatBundle')
        ])
]

# test_pipeline = [ dict(type='LoadImageFromFile'),
#                   dict( type='MultiScaleFlipAug',
#                         img_scale=(1024, 1024),
#                         flip=False,
#                         transforms=[
#                             dict(type='Resize', keep_ratio=True),
#                             dict(type='RandomFlip'),
#                             dict(type='Normalize', **img_norm_cfg),
#                             dict(type='Pad', size_divisor=32),
#                         )
#                   ]
#                 # dict(type='ImageToTensor', keys=['img']),
#                 # dict(type='DefaultFormatBundle'), dict(type='Collect', keys=['img']), ])
#                 # ]

# evaluation = dict(metric=['bbox', 'segm'])  # without this line
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)

runner = dict(type='EpochBasedRunner', max_epochs=50)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    step=[16, 22])
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])

checkpoint_config = dict(interval=3)
load_from='mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'