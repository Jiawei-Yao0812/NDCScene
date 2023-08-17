import pickle
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch

NYU_class_names = [
    "empty",
    "ceiling",
    "floor",
    "wall",
    "window",
    "chair",
    "bed",
    "sofa",
    "table",
    "tvs",
    "furn",
    "objs",
]
result_path = '/root/autodl-tmp/NDCScene_dev_head/ndcscene/exp/lr_0.1_1_head_2/output/NYU/*.pkl'
# result_path = '/root/autodl-tmp/NDCScene_dev/ndcscene/exp/exp_1/output/NYU/*.pkl'
# result_path = '/root/autodl-tmp/NDCScene_dev_coord/ndcscene/exp/trial_new_cpr_1/output/NYU/*.pkl'
result_names = glob.glob(result_path)

scene_size = (60, 36, 60)
class_num  = 12
prob = []
target = []

for i, result_name in tqdm(enumerate(result_names)):
    result = pickle.load(open(result_name, 'rb'))
    if not 'y_prob_loc' in result:
        p = torch.from_numpy(result['y_prob']).permute(1, 2, 3, 0).reshape(-1, class_num)
    else:
        p_cls = torch.from_numpy(result['y_prob_cls']).permute(1, 2, 3, 0).reshape(-1, class_num - 1)
        p_loc = torch.from_numpy(result['y_prob_loc']).permute(1, 2, 3, 0).reshape(-1, 2)
        p = torch.cat([p_loc[:,:1], p_loc[:,1:] * p_cls], dim=1)
    t = torch.from_numpy(result['target'].astype(np.int32)).reshape(-1)
    m = (t != 255)
    prob.append(p[m])
    target.append(t[m])

prob = torch.cat(prob, dim=0)
target = torch.cat(target, dim=0)
pred = prob[:,1:].argmax(dim=1) + 1

idx = (1 - prob[:,0]).argsort(dim=0, descending=True)
idx = prob[:,1:].max(dim=1).values.argsort(dim=0, descending=True)
pred = pred[idx]
target = target[idx]
rcl = 0
iou = 0
for i in range(1, class_num):
    tp = ((pred==i) & (target==i)).cumsum(dim=0)
    fp = ((pred==i) & (target!=i)).cumsum(dim=0)
    nc = (target==i).sum()
    rcl += tp / nc
    if i==1:
        print((tp / (fp + nc)).max(dim=0).values.item())
        iou += (tp / (fp + nc)).max(dim=0).values.item()
    else:
        iou += tp / (fp + nc)


rcl = (rcl / (class_num - 1)).cpu().numpy()
iou = (iou / (class_num - 1)).cpu().numpy()
print(iou.max())

plt.plot(rcl, iou)
plt.savefig('rcl_iou.png')




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