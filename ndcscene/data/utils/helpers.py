import numpy as np
import ndcscene.data.utils.fusion as fusion
import torch
import torch.nn.functional as F

def unfold(x, kernel_size, dilation, padding, stride):
    # x [b, c, h, w, d]
    b, c, h, w, d = x.shape
    x = x.reshape(b * c, h, w, d)
    x = F.unfold(x, kernel_size, dilation=dilation, padding=padding, stride=stride)
    x = x.reshape(b * c, h, kernel_size * kernel_size, w // stride, d // stride).permute(0, 2, 4, 1, 3)
    x = x.reshape(b, c * kernel_size * kernel_size * d // stride, h, w // stride)
    x = F.unfold(x, (kernel_size, 1), dilation=(dilation, 1), padding=(padding, 0), stride=(stride, 1))
    x = x.reshape(b, c * kernel_size * kernel_size, d // stride, kernel_size, h // stride * w // stride).permute(0, 1, 3, 4, 2)
    x = x.reshape(b, c * kernel_size * kernel_size * kernel_size, h // stride * w // stride * d // stride)
    return x

def compute_CP_mega_matrix(target, stride=4, is_binary=False):
    kernel_size = 9
    dilation = stride
    h, w, d = target.shape
    target = torch.from_numpy(target).reshape(1, 1, h, w, d).float()
    target_s = unfold(target, 1, dilation=1, padding=0, stride=stride).squeeze(0) # [1, h//s * w//s * d//s]
    target_t = unfold(target, kernel_size, dilation=dilation, padding=kernel_size // 2 * dilation, stride=stride).squeeze(0) # [k^3, h//s * w//s * d//s]
    target_t_mask = unfold(torch.ones_like(target), kernel_size, dilation=dilation, padding=kernel_size // 2 * dilation, stride=stride).squeeze(0) # [k^3, h//s * w//s * d//s]
    eq = (target_s == target_t)
    neq = ~eq
    if is_binary:
        matrix = torch.stack([eq, neq], dim=2)
    else:
        zero = (target_t == 0)
        nonzero = ~zero
        matrix = torch.stack([eq & zero, neq & zero, eq & nonzero, neq & nonzero], dim=2)
    valid_mask = ((target_s != 255) & (target_t != 255) & (target_t_mask > 0)).unsqueeze(2)
    matrix = torch.cat([matrix, valid_mask], dim=2)
    return matrix

def vox2pix(cam_E, cam_k, 
            vox_origin, vox_size, vox_dim, 
            img_W, img_H, img_D, img_S,
            ):
    vox_x, vox_y, vox_z = np.meshgrid(
            range(vox_dim[0]),
            range(vox_dim[1]),
            range(vox_dim[2]),
            indexing='ij'
          )
    vox_coord = np.stack([vox_x, vox_y, vox_z], axis=3).reshape(-1, 3)

    # Project voxels'centroid from lidar coordinates to camera coordinates
    cam_pt = vox_origin[None, :] + (vox_coord + 0.5) * vox_size # [N, 3]
    cam_pt = np.dot(np.hstack([cam_pt, np.ones((len(cam_pt), 1), dtype=np.float32)]), cam_E.T)[:,:3]
    cam_pt[:,2] = cam_pt[:,2] / img_S

    # Project camera coordinates to pixel positions
    f = np.array([[cam_k[0, 0], cam_k[1, 1]]])
    c = np.array([[cam_k[0, 2], cam_k[1, 2]]])
    pix = cam_pt[:,:2] * f / cam_pt[:,2:] + c
    pix_x, pix_y, pix_z = pix[:, 0], pix[:, 1], cam_pt[:,2]
    # Eliminate pixels outside view frustum
    fov_mask = np.logical_and(pix_x >= 0,
                np.logical_and(pix_x <= img_W - 1,
                np.logical_and(pix_y >= 0,
                np.logical_and(pix_y <= img_H - 1,
                np.logical_and(pix_z >= img_D[0],
                pix_z <= img_D[1])))))

    projected_pix = np.stack([pix_x, pix_y, (pix_z - img_D[0]) / (img_D[1] - img_D[0])], axis=1)

    return projected_pix, fov_mask

def compute_local_frustums(projected_pix, fov_mask, target, dataset, n_classes, size=4):
    projected_pix = projected_pix.reshape((1, -1, 2)) # [1, H * W * D, 2]
    fov_mask = fov_mask.reshape((1, -1)) # [1, H * W * D]
    frustum_x, frustum_y = np.meshgrid(np.arange(size + 1) / size, np.arange(size + 1) / size) # [size, size]
    frustum_l = np.stack([frustum_x[:-1,:-1], frustum_y[:-1,:-1]], axis=2).reshape((-1, 1, 2)) # [size * size, 1, 2]
    frustum_r = np.stack([frustum_x[:-1,1:], frustum_y[1:,:-1]], axis=2).reshape((-1, 1, 2)) # [size * size, 1, 2]
    frustum_mask = fov_mask & (projected_pix >= frustum_l).all(2) & (projected_pix < frustum_r).all(2) # [size * size, H * W * D]
    if dataset == "NYU":
        frustum_mask = np.moveaxis(frustum_mask.reshape(-1, 60, 60, 36), [0, 1, 2, 3], [0, 1, 3, 2]) # [size * size, H, W, D]
    elif dataset == "kitti":
        frustum_mask = frustum_mask.reshape(-1, *target.shape) # [size * size, H, W, D]
    frustum_mask = frustum_mask & (target != 255)[None, :, :, :] # [size * size, H, W, D]
    frustum_dist = (frustum_mask[:, :, :, :, None] & (target[None, :, :, :, None] == np.arange(n_classes).reshape(1, 1, 1, 1, -1))).reshape(size * size, -1, n_classes).sum(axis=1) # [size * size, classes]
    return frustum_mask, frustum_dist