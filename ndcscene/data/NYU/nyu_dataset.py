import torch
import os
import glob
import copy
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from ndcscene.data.utils.helpers import (
    vox2pix,
    compute_local_frustums,
    compute_CP_mega_matrix,
)
import pickle

val_imgs = ['NYU0435_0000_color.jpg', 'NYU0591_0000_color.jpg', 'NYU0917_0000_color.jpg', 'NYU0932_0000_color.jpg', 'NYU0970_0000_color.jpg', 'NYU0977_0000_color.jpg', 'NYU0995_0000_color.jpg', 'NYU1148_0000_color.jpg', 'NYU1162_0000_color.jpg', 'NYU1250_0000_color.jpg']

class NYUDataset(Dataset):
    def __init__(
        self,
        split,
        root,
        preprocess_root,
        n_relations=4,
        color_jitter=None,
        frustum_size=4,
        fliplr=0.0,
        resize=False,
    ):
        # dataset and dir
        self.n_classes = 12
        root = os.path.join(root, "NYU" + split)
        bin_root = os.path.join(preprocess_root, "base", "NYU" + split)
        scan_names = glob.glob(os.path.join(root, "*.bin"))
        scan_names = [file_path for file_path in scan_names if (os.path.basename(file_path)[:-4] + "_color.jpg" in val_imgs)]
        self.rgb_names = [os.path.join(root, os.path.basename(file_path)[:-4] + "_color.jpg") for file_path in scan_names]
        self.pkl_names = [os.path.join(bin_root, os.path.basename(file_path)[:-4] + ".pkl") for file_path in scan_names]
        # 2d img
        self.resize = resize
        self.resize_list = [(320, 240), (480, 360), (640, 480), (800, 600), (960, 720)]
        self.resize_prob = [0.125, 0.125, 0.5, 0.125, 0.125]
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.fliplr = fliplr
        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # 3d space
        self.vox_size = 0.08  # 0.08m
        self.vox_dim = (60, 60, 36)  # (4.8m, 4.8m, 2.88m)
        self.img_W = 640
        self.img_H = 480
        self.img_D = [0.5538, 6.8243]
        self.cam_k = np.array([[518.8579, 0, 320], [0, 518.8579, 240], [0, 0, 1]])
        # algorithm parameters
        self.n_relations = n_relations
        self.frustum_size = frustum_size

    def __getitem__(self, index):
        with open(self.pkl_names[index], "rb") as handle:
            data = pickle.load(handle)
        # compute the 3D-2D mapping
        img_WH = (640, 480)
        img_S = 1.0
        if self.resize:
            i = np.random.choice(len(self.resize_list), p=self.resize_prob)
            img_WH = self.resize_list[i]
            img_S = img_WH[0] / 640.0
        cam_pose = data["cam_pose"]
        T_world_2_cam = np.linalg.inv(cam_pose)
        vox_origin = data["voxel_origin"]
        projected_pix, fov_mask = vox2pix(
            T_world_2_cam,
            self.cam_k,
            vox_origin,
            self.vox_size,
            self.vox_dim,
            self.img_W,
            self.img_H,
            self.img_D,
            img_S
        )
        data["cam_k"] = self.cam_k
        data["projected_pix_1"] = projected_pix
        data["fov_mask_1"] = fov_mask
        target = data["target_1_4"]
          # Following SSC literature, the output resolution on NYUv2 is set to 1:4
        if img_S != 1.0:
            target[~np.moveaxis(fov_mask.reshape(self.vox_dim), [0, 1, 2], [0, 2, 1])] = 255
        data["target"] = target

        # load CP_mega_matrix into data
        # target_1_4 = data["target_1_16"]
        CP_mega_matrix = compute_CP_mega_matrix(target, is_binary=(self.n_relations == 2), stride=4)
        data["CP_mega_matrix"] = CP_mega_matrix

        # compute the masks, each indicates voxels inside a frustum
        frustums_masks, frustums_class_dists = compute_local_frustums(
            projected_pix[:,:2] / np.array([[self.img_W, self.img_H]]),
            fov_mask,
            target,
            dataset="NYU",
            n_classes=12,
            size=self.frustum_size,
        )
        data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists

        # load rgb image
        img = Image.open(self.rgb_names[index]).convert("RGB")
        # image augmentation
        if img_WH != (640, 480):
            img = img.resize(img_WH, Image.BILINEAR)
        if img_WH[0] > 640:
            w, h = (img_WH[0] - self.img_W) // 2, (img_WH[1] - self.img_H) // 2
            img = img.crop((w, h, w + self.img_W, h + self.img_H))
        elif img_WH[0] < 640:
            w, h = (self.img_W - img_WH[0]) // 2, (self.img_H - img_WH[1]) // 2
            img_ = Image.new('RGB', (640, 480), (123, 116, 103))
            img_.paste(img, (w, h))
            img = img_

        if self.color_jitter is not None:
            img = self.color_jitter(img)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        # randomly fliplr the image
        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            data["projected_pix_1"][:, 0] = self.img_W - 1 - data["projected_pix_1"][:, 0]
        data["img"] = self.normalize_rgb(img)  # (3, img_H, img_W)

        return data

    def __len__(self):
        return len(self.rgb_names)
