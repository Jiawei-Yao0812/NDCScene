from kitti_dataset import KittiDataset
from ndcscene.data.semantic_kitti.kitti_dataset import KittiDataset
import time
import threading
import functools
dataset = KittiDataset(
    split="train",
    root='/root/autodl-tmp/DATA/NDCscene_SemanticKitti',
    preprocess_root='/root/autodl-tmp/DATA/NDCscene_SemanticKitti/preprocess',
    frustum_size=8,
    fliplr=0.5,
    color_jitter=(0.4, 0.4, 0.4),
)
print('total sample', len(dataset))

def pregetitem(dataset, be, st):
    for i in range(be, len(dataset), st):
        t0 = time.time()
        dataset.pregetitem(i)
        t1 = time.time()
        print('sample {} time cost: {:.4f}'.format(i, t1 - t0))

thread_num = 32
threads = []
for i in range(thread_num):
    threads.append(threading.Thread(target=functools.partial(pregetitem, dataset, i, thread_num)))
for t in threads:
    t.start()