from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class M2nistDataset(BaseSegDataset):
  classes = ('background', 'digits')
  # palette = [[128, 128, 128], [151, 189, 8]]
  palette = [[255, 255, 255], [[0, 0, 0]]]

  METAINFO = dict(classes = classes, palette = palette)
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)