import os.path as osp
import mmcv

# https://mmdetection.readthedocs.io/en/latest/2_new_data_model.html
# https://blog.csdn.net/weixin_44966641/article/details/118686130
def convert_balloon_to_coco(ann_file, out_file, image_prefix):
    data_infos = mmcv.load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(mmcv.track_iter_progress(data_infos.values())):
        filename = v['filename']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        bboxes = []
        labels = []
        masks = []
        for _, obj in v['regions'].items():
            assert not obj['region_attributes']
            obj = obj['shape_attributes']
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (
                min(px), min(py), max(px), max(py))


            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id':0, 'name': 'balloon'}])

    # "categories": [{"id": 0, "name": "balloon"}],
    # add this to val/train.json
    # Traning on coco2014 got KeyError: 'categories' · Issue #171 · microsoft/SoftTeacher · GitHub
    # https://github.com/microsoft/SoftTeacher/issues/171
    #
    # Training on custom dataset error: KeyError: 'SemiDataset: "CocoDataset: \'file_name\'"' · Issue #105 · microsoft/SoftTeacher · GitHub
    # https://github.com/microsoft/SoftTeacher/issues/105
    #
    # 目标检测篇：MMDetection 推理使用与详细踩坑记录 | Just for Life.
    # https://muyuuuu.github.io/2021/05/11/MMDetection-use/


    mmcv.dump(coco_format_json, out_file, indent=4)  ##, indent=4



convert_balloon_to_coco('data/balloon/train/via_region_data.json', 'data/balloon/train/train.json', r'data/balloon/train')
convert_balloon_to_coco('data/balloon/val/via_region_data.json', 'data/balloon/val/val.json', r'data/balloon/val')


# /data/balloon/train/via-region_data.json

# [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 61/61, 56.9 task/s, elapsed: 1s, ETA:     0s
# [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 13/13, 55.1 task/s, elapsed: 0s, ETA:     0s
