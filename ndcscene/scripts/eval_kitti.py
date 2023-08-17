import pickle
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch

kitti_class_names = [
    "empty",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]
result_path = '/root/autodl-tmp/NDCScene_dev_coord/ndcscene/exp/kitti_lr_0.1_1_xybr_1/output_base/kitti/08/*.pkl'
# result_path = '/root/autodl-tmp/NDCScene_dev/ndcscene/exp/exp_1/output/NYU/*.pkl'
# result_path = '/root/autodl-tmp/NDCScene_dev_coord/ndcscene/exp/trial_new_cpr_1/output/NYU/*.pkl'
result_names = glob.glob(result_path)

class_num  = 20
tp_1 = [0 for i in range(class_num)]
fp_1 = [0 for i in range(class_num)]
fn_1 = [0 for i in range(class_num)]
iou_1 = [0 for i in range(class_num)]

tp_2 = [0 for i in range(class_num)]
fp_2 = [0 for i in range(class_num)]
fn_2 = [0 for i in range(class_num)]
iou_2 = [0 for i in range(class_num)]

for result_name in tqdm(result_names):
    result = pickle.load(open(result_name, 'rb'))
    pred = torch.from_numpy(result['y_pred'].astype(np.int32)).reshape(-1)
    target = torch.from_numpy(result['target'].astype(np.int32)).reshape(-1)
    fov_mask = torch.from_numpy(result['fov_mask_1'].astype(np.bool)).reshape(-1)
    m = (target != 255) # & fov_mask
    p = pred[m]
    t = target[m]
    for i in range(class_num):
        tp_1[i] += ((p==i) & (t==i)).sum()
        fp_1[i] += ((p==i) & (t!=i)).sum()
        fn_1[i] += ((p!=i) & (t==i)).sum()

    m = m & fov_mask
    p = pred[m]
    t = target[m]
    for i in range(class_num):
        tp_2[i] += ((p==i) & (t==i)).sum()
        fp_2[i] += ((p==i) & (t!=i)).sum()
        fn_2[i] += ((p!=i) & (t==i)).sum()

for i in range(class_num):
    iou_1[i] = tp_1[i] / (tp_1[i] + fp_1[i] + fn_1[i])
    iou_2[i] = tp_2[i] / (tp_2[i] + fp_2[i] + fn_2[i])
    
print('iou_all', np.array(iou_1)[1:].mean())
print('iou_fov', np.array(iou_2)[1:].mean())






# plt.hist(prob_o[prob_o>0.5], bins=100)
# plt.savefig('occ_prob.png')

# # pred = prob[:,1:].argmax(dim=1) + 1
# # pred *= (prob[:,0] < 0.5)

# pred = prob.argmax(dim=1)
# target = target

# iou = []
# prc = []
# rcl = []
# for i in range(12):
#     tp = ((pred==i) & (target==i)).sum()
#     fp = ((pred==i) & (target!=i)).sum()
#     fn = ((pred!=i) & (target==i)).sum()
#     iou.append(tp / (tp + fp + fn))
#     prc.append(tp / (tp + fp))
#     rcl.append(tp / (tp + fn))

# res = 'miou:{:.5f}'.format(np.array(iou)[1:].mean())
# for i in range(1, 12):
#     name = NYU_class_names[i]
#     res += ',\tiou_{}:{:.5f}'.format(name, iou[i])
# print(res)

# res = 'mprc:{:.5f}'.format(np.array(prc)[1:].mean())
# for i in range(1, 12):
#     name = NYU_class_names[i]
#     res += ',\tprc_{}:{:.5f}'.format(name, prc[i])
# print(res)

# res = 'mrcl:{:.5f}'.format(np.array(rcl)[1:].mean())
# for i in range(1, 12):
#     name = NYU_class_names[i]
#     res += ',\trcl_{}:{:.5f}'.format(name, rcl[i])
# print(res)

# np.savetxt('coord.csv', np.array([np.array(iou)[1:].mean(), ] + iou + [np.array(prc)[1:].mean(), ] + prc + [np.array(rcl)[1:].mean(), ] + rcl))

# pred = prob.argmax(dim=1)
# tp = ((pred>=1) & (target>=1)).sum()
# fp = ((pred>=1) & (target==0)).sum()
# fn = ((pred==0) & (target>=1)).sum()
# iou = tp / (tp + fp + fn)

# mask = (pred>=1) & (target>=1)
# pred = pred[mask]
# target = target[mask]
# prc = []
# for i in range(1, 12):
#     p = ((pred==i) & (target==i)).sum() / (target==i).sum()
#     prc.append(p)
# res = 'iou:{:.5f},\tprc:{:.5f}'.format(iou, np.array(prc).mean())
# for i in range(1, 12):
#     name = NYU_class_names[i]
#     res += ',\tprc_{}:{:.5f}'.format(name, prc[i-1])
# print(res)