from pytorch_lightning import Trainer
from ndcscene.models.ndcscene import NDCScene
from ndcscene.data.NYU.nyu_dm import NYUDataModule
from ndcscene.data.semantic_kitti.kitti_dm import KittiDataModule
from ndcscene.data.kitti_360.kitti_360_dm import Kitti360DataModule
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import os
from hydra.utils import get_original_cwd
from tqdm import tqdm
import pickle


@hydra.main(config_name="../config/ndcscene_nyu.yaml")
def main(config: DictConfig):
    for k, v in config.items():
        if v == 'None' or v == 'none':
            config[k] = None
    config.n_gpus = 1

    torch.set_grad_enabled(False)

    # Setup dataloader
    if config.dataset == "kitti" or config.dataset == "kitti_360":
        feature = 64
        project_scale = 2
        full_scene_size = (256, 256, 32)

        if config.dataset == "kitti":
            data_module = KittiDataModule(
                root=config.kitti_root,
                preprocess_root=config.kitti_preprocess_root,
                frustum_size=config.frustum_size,
                batch_size=int(config.batch_size / config.n_gpus),
                num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            )
            data_module.setup()
            data_loader = data_module.val_dataloader()
            # data_loader = data_module.test_dataloader() # use this if you want to infer on test set
        else:
            data_module = Kitti360DataModule(
                root=config.kitti_360_root,
                sequences=[config.kitti_360_sequence],
                n_scans=2000,
                batch_size=1,
                num_workers=3,
            )
            data_module.setup()
            data_loader = data_module.dataloader()

    elif config.dataset == "NYU":
        project_scale = 1
        feature = 200
        full_scene_size = (60, 36, 60)
        data_module = NYUDataModule(
            root=config.NYU_root,
            preprocess_root=config.NYU_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            resize=config.resize,
        )
        data_module.setup()
        data_loader = data_module.val_dataloader()
        # data_loader = data_module.test_dataloader() # use this if you want to infer on test set
    else:
        print("dataset not support")

    # Load pretrained models
    if config.dataset == "NYU":
        model_path = os.path.join(
            get_original_cwd(), "trained_models", "ndcscene_nyu.ckpt"
        )
    else:
        model_path = os.path.join(
            get_original_cwd(), "trained_models", "ndcscene_kitti.ckpt"
        )

    # model_path = '/root/autodl-tmp/NDCScene_dev_coord/ndcscene/exp/trial_new_cpr_1/exp_NYU_1_FrusSize_8_nRelations4_WD0.0001_lr0.0001_CEssc_geoScalLoss_semScalLoss_fpLoss_CERel_3DCRP/checkpoints/epoch=024-val/mIoU=0.26861.ckpt'
    # model_path = '/root/autodl-tmp/NDCScene_dev_coord/ndcscene/exp/kitti_lr_0.1_1_xybr_1/exp_kitti_1_FrusSize_8_nRelations4_WD0.0001_lr0.0001_CEssc_geoScalLoss_semScalLoss_fpLoss_CERel_3DCRP/checkpoints/epoch=025-val/mIoU=0.10857.ckpt'
    # model_path = '/root/autodl-tmp/NDCScene_dev_coord/ndcscene/exp/kitti_lr_0.1_1_xybr_full_1/exp_kitti_1_FrusSize_8_nRelations4_WD0.0001_lr0.0001_CEssc_geoScalLoss_semScalLoss_fpLoss_CERel_3DCRP/checkpoints/epoch=026-val/mIoU=0.12696.ckpt'
    model_path = '/root/autodl-tmp/NDCScene_dev_coord/ndcscene/exp/trial_lr_0.1_1_xybr_3/exp_NYU_1_FrusSize_8_nRelations4_WD0.0001_lr0.0001_CEssc_geoScalLoss_semScalLoss_fpLoss_CERel_3DCRP/checkpoints/epoch=020-val/mIoU=0.29029.ckpt'
    model = NDCScene.load_from_checkpoint(
        model_path,
        feature=feature,
        project_scale=project_scale,
        fp_loss=config.fp_loss,
        full_scene_size=full_scene_size,
    )
    model.cuda()
    model.eval()

    # Save prediction and additional data 
    # to draw the viewing frustum and remove scene outside the room for NYUv2
    output_path = os.path.join(config.output_path, config.dataset)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch["img"] = batch["img"].cuda()
            pred = model(batch)
            y_prob = torch.softmax(pred["ssc_logit"], dim=1).detach().cpu().numpy()
            y_pred = np.argmax(y_prob, axis=1)
            for i in range(batch["img"].shape[0]):
                out_dict = {"y_pred": y_pred[i].astype(np.uint16), "y_prob": y_prob[i].astype(np.float32)}
                if "target" in batch:
                    out_dict["target"] = (
                        batch["target"][i].detach().cpu().numpy().astype(np.uint16)
                    )

                if config.dataset == "NYU":
                    write_path = output_path
                    filepath = os.path.join(write_path, batch["name"][i] + ".pkl")
                    out_dict["cam_pose"] = batch["cam_pose"][i].detach().cpu().numpy()
                    out_dict["vox_origin"] = (
                        batch["vox_origin"][i].detach().cpu().numpy()
                    )
                else:
                    write_path = os.path.join(output_path, batch["sequence"][i])
                    filepath = os.path.join(write_path, batch["frame_id"][i] + ".pkl")
                    out_dict["fov_mask_1"] = (
                        batch["fov_mask_1"][i].detach().cpu().numpy()
                    )
                    out_dict["cam_k"] = batch["cam_k"][i].detach().cpu().numpy()
                    out_dict["T_velo_2_cam"] = (
                        batch["T_velo_2_cam"][i].detach().cpu().numpy()
                    )

                os.makedirs(write_path, exist_ok=True)
                with open(filepath, "wb") as handle:
                    pickle.dump(out_dict, handle)
                    print("wrote to", filepath)


if __name__ == "__main__":
    main()
