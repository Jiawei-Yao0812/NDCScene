import os
import glob
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from ndcscene.data.utils.helpers import (
    vox2pix,
    compute_local_frustums,
    compute_CP_mega_matrix,
)

val_imgs = ['000085', '000290', '000790', '001385', '001500', '001530', '002505', '003190', '003420', '003790', '000295']

class KittiDataset(Dataset):
    def __init__(
        self,
        split,
        root,
        preprocess_root,
        frustum_size=4,
        color_jitter=None,
        fliplr=0.0,
    ):
        super().__init__()
        # dataset
        self.n_classes = 20
        self.root = root
        self.label_root = os.path.join(preprocess_root, "labels")
        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }
        self.split = split
        self.sequences = splits[split]
        self.scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.root, "dataset", "sequences", sequence, "calib.txt")
            )
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix = P @ T_velo_2_cam

            glob_path = os.path.join(
                self.root, "dataset", "sequences", sequence, "voxels", "*.bin"
            )
            for voxel_path in glob.glob(glob_path):
                if split!= 'val' or os.path.splitext(os.path.basename(voxel_path))[0] in val_imgs:
                    self.scans.append(
                        {
                            "sequence": sequence,
                            "P": P,
                            "T_velo_2_cam": T_velo_2_cam,
                            "proj_matrix": proj_matrix,
                            "voxel_path": voxel_path,
                        }
                    )
        # 2/3d parameters
        self.vox_size = 0.2  # 0.2m
        self.vox_dim = np.array([256, 256, 32])
        self.vox_origin = np.array([0, -25.6, -2])
        self.img_W = 1220
        self.img_H = 370
        self.img_D = [0.2, 50.8]
        # 2d aug
        self.fliplr = fliplr
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # algo parameters
        self.frustum_size = frustum_size

        # self.scans = self.scans[::5]

    def __getitem__(self, index):
        # if self.split == 'train':
        #     return self.getitem_fast(index)
        # get file info
        scan = self.scans[index]
        sequence = scan["sequence"]
        voxel_path = scan["voxel_path"]
        frame_id = os.path.splitext(os.path.basename(voxel_path))[0]
        rgb_path = os.path.join(
            self.root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
        )
        # get 3d data
        P = scan["P"]
        T_velo_2_cam = scan["T_velo_2_cam"]
        proj_matrix = scan["proj_matrix"]
        data = {
            "frame_id": frame_id,
            "sequence": sequence,
            "P": P,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix": proj_matrix,
        }
        data["scale_3ds"] = [1, 2]
        cam_k = P[0:3, 0:3]
        data["cam_k"] = cam_k
        # compute the 3D-2D mapping
        projected_pix_1, fov_mask_1 = vox2pix(
            T_velo_2_cam,
            cam_k,
            self.vox_origin,
            self.vox_size,
            self.vox_dim.astype(np.int32),
            self.img_W,
            self.img_H,
            self.img_D,
            1
        )
        # print(projected_pix_1[:,0].min(), projected_pix_1[:,0].max())
        projected_pix_2, fov_mask_2 = vox2pix(
            T_velo_2_cam,
            cam_k,
            self.vox_origin,
            self.vox_size * 2,
            (self.vox_dim / 2).astype(np.int32),
            self.img_W,
            self.img_H,
            self.img_D,
            1
        )
        data["projected_pix_1"] = projected_pix_1
        data["projected_pix_2"] = projected_pix_2
        data["fov_mask_1"] = fov_mask_1
        data["fov_mask_2"] = fov_mask_2

        target_1_path = os.path.join(self.label_root, sequence, frame_id + "_1_1.npy")
        target = np.load(target_1_path)
        data["target"] = target
        target_8_path = os.path.join(self.label_root, sequence, frame_id + "_1_8.npy")
        target_1_8 = np.load(target_8_path)
        CP_mega_matrix = compute_CP_mega_matrix(target_1_8, stride=1)
        data["CP_mega_matrix"] = CP_mega_matrix

        # Compute the masks, each indicate the voxels of a local frustum
        if self.split != "test":
            frustums_masks, frustums_class_dists = compute_local_frustums(
                projected_pix_1[:,:2] / np.array([[self.img_W, self.img_H]]),
                fov_mask_1,
                target,
                dataset="kitti",
                n_classes=20,
                size=self.frustum_size,
            )
        else:
            frustums_masks = None
            frustums_class_dists = None
        data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists

        # get 2d data
        img = Image.open(rgb_path).convert("RGB")
        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        img = img[:self.img_H, :self.img_W, :]  # crop image
        # Fliplr the image
        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            data["projected_pix_1"][:, 0] = self.img_W - 1 - data["projected_pix_1"][:, 0]
            data["projected_pix_2"][:, 0] = self.img_W - 1 - data["projected_pix_2"][:, 0]
        # Normalize the image
        data["img"] = self.normalize_rgb(img)

        return data

    def __len__(self):
        return len(self.scans)

    @staticmethod
    def read_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out

    def pregetitem(self, index):
        # get file info
        scan = self.scans[index]
        sequence = scan["sequence"]
        voxel_path = scan["voxel_path"]
        frame_id = os.path.splitext(os.path.basename(voxel_path))[0]
        
        data = self.__getitem__(index)
        data.pop('img')
        pkl_path = os.path.join(self.label_root, sequence, frame_id + ".pkl")
        print('dump: ' + pkl_path)
        pickle.dump(data, open(pkl_path, 'wb'))

    def getitem_fast(self, index):
        # get file info
        scan = self.scans[index]
        sequence = scan["sequence"]
        voxel_path = scan["voxel_path"]
        frame_id = os.path.splitext(os.path.basename(voxel_path))[0]
        rgb_path = os.path.join(
            self.root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
        )

        pkl_path = os.path.join(self.label_root, sequence, frame_id + ".pkl")
        data = pickle.load(open(pkl_path, 'rb'))

        # get 2d data
        img = Image.open(rgb_path).convert("RGB")
        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        img = img[:self.img_H, :self.img_W, :]  # crop image
        # Fliplr the image
        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            data["projected_pix_1"][:, 0] = self.img_W - 1 - data["projected_pix_1"][:, 0]
            data["projected_pix_2"][:, 0] = self.img_W - 1 - data["projected_pix_2"][:, 0]
        # Normalize the image
        data["img"] = self.normalize_rgb(img)

        return data