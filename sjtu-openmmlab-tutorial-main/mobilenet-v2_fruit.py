model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=30,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))

load_from = 'mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'

# dataset_type = 'ImageNet'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='RandomResizedCrop', size=224, backend='pillow'),
#     dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
#     dict(
#         type='Normalize',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         to_rgb=True),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', size=(256, -1), backend='pillow'),
#     dict(type='CenterCrop', crop_size=224),
#     dict(
#         type='Normalize',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         to_rgb=True),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='Collect', keys=['img'])
# ]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        # type='ImageNet',
        type='CustomDataset',

        # data_prefix='data/imagenet/train',
        data_prefix='data/fruit30_split/train',

        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=224, backend='pillow'),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        # type='ImageNet',
        type='CustomDataset',
        # data_prefix='data/imagenet/val',
        data_prefix='data/fruit30_split/val',
        # ann_file='data/imagenet/meta/val.txt',  ##### 可以删除或者None
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1), backend='pillow'),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        # type='ImageNet',
        type='CustomDataset',
        data_prefix='data/fruit30_split/val',
        # ann_file='data/imagenet/meta/val.txt',  ##### 可以删除或者None
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1), backend='pillow'),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))

evaluation = dict(interval=1, metric='accuracy')
# optimizer = dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=4e-05)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=4e-05)
# 8卡 0.045, 除以8,然后微调需要更小

optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', gamma=0.98, step=1)
# runner = dict(type='EpochBasedRunner', max_epochs=300)
runner = dict(type='EpochBasedRunner', max_epochs=5)
# checkpoint_config = dict(interval=1)
# log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
checkpoint_config = dict(interval=5)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])

dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = None    ######预训练权重路径

resume_from = None
workflow = [('train', 1)]
