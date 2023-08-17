from kitti_dataset import KittiDataset
from ndcscene.data.semantic_kitti.kitti_dataset import KittiDataset
import time
import threading
import functools
split = 'val'
dataset_1 = KittiDataset(
    split=split,
    root='/root/autodl-tmp/DATA/NDCscene_SemanticKitti',
    preprocess_root='/root/autodl-tmp/DATA/NDCscene_SemanticKitti/preprocess',
    frustum_size=8,
    fliplr=0.5,
    color_jitter=(0.4, 0.4, 0.4),
)

dataset_2 = KittiDataset(
    split=split,
    root='/root/autodl-tmp/DATA/NDCscene_SemanticKitti',
    preprocess_root='/root/autodl-tmp/DATA/NDCscene_SemanticKitti/preprocess',
    frustum_size=8,
    fliplr=0.5,
    color_jitter=(0.4, 0.4, 0.4),
)
print('total sample', len(dataset_1))

for i in range(20):
    assert dataset_1.scans[i]['voxel_path'] == dataset_2.scans[i]['voxel_path'], print(dataset_1.scans[i]['voxel_path'], dataset_2.scans[i]['voxel_path'])

