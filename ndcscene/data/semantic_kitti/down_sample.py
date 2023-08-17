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

class KittiDataset(Dataset):
    def __init__(
        self,
        root,
        preprocess_root,
        split='full',
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
            "full": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
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

    def __getitem__(self, index):
        scan = self.scans[index]
        sequence = scan["sequence"]
        voxel_path = scan["voxel_path"]
        frame_id = os.path.splitext(os.path.basename(voxel_path))[0]

        return os.path.join(self.label_root, sequence, frame_id)

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

def downsample_label(label, ds=4):
    r"""downsample the labeled data,
    code taken from https://github.com/waterljwant/SSC/blob/master/dataloaders/dataloader.py#L262
    Shape:
        label, (240, 144, 240)
        label_downscale, if ds==4, then (60, 36, 60)
    """
    voxel_size = label.shape
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )  # small size
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    label = np.pad(label, (ds // 2, 0), 'constant', constant_values=(255, 255))
    empty_t = 0.95 * (ds + 1) * (ds + 1) * (ds + 1)  # threshold
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds + 1, ds + 1, ds + 1), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds + 1, y * ds : (y + 1) * ds + 1, z * ds : (z + 1) * ds + 1
        ]
        label_bin = label_i.flatten()

        zero_count_0 = np.array(np.where(label_bin == 0)).size
        zero_count_255 = np.array(np.where(label_bin == 255)).size

        zero_count = zero_count_0 + zero_count_255
        if zero_count > empty_t:
            label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
        else:
            label_i_s = label_bin[
                np.where(np.logical_and(label_bin > 0, label_bin < 255))
            ]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    return label_downscale

def downsample_label_ori(label, voxel_size=(240, 144, 240), downscale=4):
    r"""downsample the labeled data,
    code taken from https://github.com/waterljwant/SSC/blob/master/dataloaders/dataloader.py#L262
    Shape:
        label, (240, 144, 240)
        label_downscale, if downsample==4, then (60, 36, 60)
    """
    if downscale == 1:
        return label
    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )  # small size
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 0.95 * ds * ds * ds  # threshold
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds, y * ds : (y + 1) * ds, z * ds : (z + 1) * ds
        ]
        label_bin = label_i.flatten()

        zero_count_0 = np.array(np.where(label_bin == 0)).size
        zero_count_255 = np.array(np.where(label_bin == 255)).size

        zero_count = zero_count_0 + zero_count_255
        if zero_count > empty_t:
            label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
        else:
            label_i_s = label_bin[
                np.where(np.logical_and(label_bin > 0, label_bin < 255))
            ]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    return label_downscale

dataset = KittiDataset(
    split="full",
    root='/root/autodl-tmp/DATA/NDCscene_SemanticKitti',
    preprocess_root='/root/autodl-tmp/DATA/NDCscene_SemanticKitti/preprocess',
    frustum_size=8,
    fliplr=0.5,
    color_jitter=(0.4, 0.4, 0.4),
)

def process(target_path):
    target_1_path = target_path + "_1_1.npy"
    target_8_path_fix = target_path + "_1_8_fix.npy"
    target_8_path_ori = target_path + "_1_8.npy"

    target_1 = np.load(target_1_path)
    target_8_ori = np.load(target_path + "_1_8.npy")
    target_8_fix = downsample_label(target_1, ds=8)
    target_8_chk = downsample_label_ori(target_1, voxel_size=(256, 256, 32), downscale=8)
    print((target_8_ori==target_8_chk).all())
    print((target_8_ori==255).sum() / target_8_fix.size, (target_8_fix==255).sum() / target_8_fix.size)
    print((target_8_ori==0).sum() / target_8_fix.size, (target_8_fix==0).sum() / target_8_fix.size)
    print((target_8_ori==target_8_fix).sum() / target_8_fix.size)
    # print((target_1[::8,::8,::8]==target_8).sum() / (32 * 32 * 4))
    # np.save(target_8_path, target_8)

for i in range(1):
    target_path = dataset[i]
    process(target_path)
    print('sample {} processed'.format(i))
