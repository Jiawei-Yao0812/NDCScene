import pytorch_lightning as pl
import torch
import torch.nn as nn
from ndcscene.models.unet3d_nyu import UNet3D as UNet3DNYU
from ndcscene.models.unet3d_kitti import UNet3D as UNet3DKitti
from ndcscene.loss.sscMetrics import SSCMetrics
from ndcscene.loss.ssc_loss import sem_scal_loss, CE_ssc_loss, KL_sep, geo_scal_loss, frustum_loss, miou_loss
from ndcscene.models.flosp import FLoSP
from ndcscene.loss.CRP_loss import compute_super_CP_multilabel_loss
import numpy as np
import torch.nn.functional as F
from ndcscene.models.unet2d import UNet2D
from ndcscene.models.modules import SegmentationHead, LightSegmentationHead
from torch.optim.lr_scheduler import MultiStepLR


class NDCScene(pl.LightningModule):
    def __init__(
        self,
        n_classes,
        class_names,
        basemodel_name,
        feature,
        class_weights,
        project_scale,
        full_scene_size,
        dataset,
        n_relations=4,
        context_prior=True,
        fp_loss=True,
        frustum_size=4,
        relation_loss=False,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        aux_ssc_loss=False,
        miou_loss=False,
        lr=1e-4,
        weight_decay=1e-4,
        milestones=[20, ],
    ):
        super().__init__()
        # dataset params
        self.dataset = dataset
        self.n_classes = n_classes
        self.class_names = class_names
        self.project_scale = project_scale
        # alg params
        self.context_prior = context_prior
        self.frustum_size = frustum_size
        self.class_weights = class_weights
        # loss params
        self.relation_loss = relation_loss
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.fp_loss = fp_loss
        self.aux_ssc_loss = aux_ssc_loss
        self.miou_loss = miou_loss
        # optim params
        self.lr = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        if self.dataset == "NYU":
            self.net_rgb = UNet2D.build(basemodel_name, out_feature=feature, output_padding=[[1, 1], [1, 1], [1, 1]])
        if self.dataset == "kitti":
            self.net_rgb = UNet2D.build(basemodel_name, out_feature=feature, output_padding=[[1, 0], [0, 0], [0, 0]])
        self.project = FLoSP(full_scene_size, project_scale=self.project_scale, dataset=self.dataset, feature=feature)
        if aux_ssc_loss:
            self.aux_head = LightSegmentationHead(feature, feature, self.n_classes, [1, 3])
        if self.dataset == "NYU":
            self.net_3d_decoder = UNet3DNYU(
                self.n_classes,
                nn.BatchNorm3d,
                n_relations=n_relations,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
            )
        elif self.dataset == "kitti":
            self.net_3d_decoder = UNet3DKitti(
                self.n_classes,
                nn.BatchNorm3d,
                project_scale=project_scale,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
            )

        # log hyperparameters
        self.save_hyperparameters()

        self.train_metrics = SSCMetrics(self.n_classes)
        self.val_metrics = SSCMetrics(self.n_classes)
        self.test_metrics = SSCMetrics(self.n_classes)

    def forward(self, batch):
        img = batch["img"]
        projected_pix = batch["projected_pix_{}".format(self.project_scale)].cuda()
        fov_mask = batch["fov_mask_{}".format(self.project_scale)].cuda()
        # get 2D feature
        x_rgb = self.net_rgb(img)
        # project features at each 2D scale to target 3D scale
        x3d = self.project(x_rgb, projected_pix, fov_mask)
        # get result
        out_dict = self.net_3d_decoder({"x3d": x3d})
        # get aux result
        if self.aux_ssc_loss:
            out_dict['aux_ssc_logit'] = self.aux_head(x3d)
        return out_dict

    def step(self, batch, step_type, metric):
        bs = len(batch["img"])
        loss = 0
        out_dict = self(batch)
        ssc_pred = out_dict["ssc_logit"]
        target = batch["target"]

        if self.context_prior and self.relation_loss:
            P_logits = out_dict["P_logits"]
            CP_mega_matrices = batch["CP_mega_matrices"]
            loss_rel_ce = compute_super_CP_multilabel_loss(
                P_logits, CP_mega_matrices
            )
            loss += loss_rel_ce
            self.log(
                step_type + "/loss_relation_ce_super",
                loss_rel_ce.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        class_weight = self.class_weights.type_as(batch["img"])
        if self.CE_ssc_loss:
            loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
            loss += loss_ssc
            self.log(
                step_type + "/loss_ssc",
                loss_ssc.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.sem_scal_loss:
            loss_sem_scal = sem_scal_loss(ssc_pred, target)
            loss += loss_sem_scal
            self.log(
                step_type + "/loss_sem_scal",
                loss_sem_scal.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.geo_scal_loss:
            loss_geo_scal = geo_scal_loss(ssc_pred, target)
            loss += loss_geo_scal
            self.log(
                step_type + "/loss_geo_scal",
                loss_geo_scal.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.fp_loss and step_type != "test":
            frustums_masks = torch.stack(batch["frustums_masks"])
            frustums_class_dists = torch.stack(
                batch["frustums_class_dists"]
            ).float()  # (bs, n_frustums, n_classes)
            loss_frustum = frustum_loss(ssc_pred, frustums_masks, frustums_class_dists)
            loss += loss_frustum
            self.log(
                step_type + "/loss_frustums",
                loss_frustum.detach(),
                on_epoch=True,
                sync_dist=True,
            )
        if self.aux_ssc_loss:
            loss_aux_ssc = CE_ssc_loss(out_dict["aux_ssc_logit"], target, class_weight)
            loss += 0.5 * loss_aux_ssc
            self.log(
                step_type + "/loss_aux_ssc",
                loss_aux_ssc.detach(),
                on_epoch=True,
                sync_dist=True,
            )
        if self.miou_loss:
            loss_miou = miou_loss(ssc_pred, target)
            loss += loss_miou
            self.log(
                step_type + "/loss_miou",
                loss_miou.detach(),
                on_epoch=True,
                sync_dist=True,
            )
        y_true = target.cpu().numpy()
        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        metric.add_batch(y_pred, y_true)

        self.log(step_type + "/loss", loss.detach(), on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", self.val_metrics)

    def validation_epoch_end(self, outputs):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]

        for prefix, metric in metric_list:
            stats = metric.get_stats()
            for i, class_name in enumerate(self.class_names):
                self.log(
                    "{}_SemIoU/{}".format(prefix, class_name),
                    stats["iou_ssc"][i],
                    sync_dist=True,
                )
            self.log("{}/mIoU".format(prefix), stats["iou_ssc_mean"], sync_dist=True)
            self.log("{}/IoU".format(prefix), stats["iou"], sync_dist=True)
            self.log("{}/Precision".format(prefix), stats["precision"], sync_dist=True)
            self.log("{}/Recall".format(prefix), stats["recall"], sync_dist=True)
            metric.reset()

    def test_step(self, batch, batch_idx):
        self.step(batch, "test", self.test_metrics)

    def test_epoch_end(self, outputs):
        classes = self.class_names
        metric_list = [("test", self.test_metrics)]
        for prefix, metric in metric_list:
            print("{}======".format(prefix))
            stats = metric.get_stats()
            print(
                "Precision={:.4f}, Recall={:.4f}, IoU={:.4f}".format(
                    stats["precision"] * 100, stats["recall"] * 100, stats["iou"] * 100
                )
            )
            print("class IoU: {}, ".format(classes))
            print(
                " ".join(["{:.4f}, "] * len(classes)).format(
                    *(stats["iou_ssc"] * 100).tolist()
                )
            )
            print("mIoU={:.4f}".format(stats["iou_ssc_mean"] * 100))
            metric.reset()

    def configure_optimizers(self):
        encoder_params = []
        other_params = []
        for k, p in self.named_parameters():
            if 'encoder' in k:
                encoder_params.append(p)
            else:
                other_params.append(p)
        params_list = [{'params': encoder_params, 'lr': self.lr * 0.1},
                   {'params': other_params}]

        if self.dataset == "NYU":
            optimizer = torch.optim.AdamW(
                params_list, lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=0.1)
            return [optimizer], [scheduler]
        elif self.dataset == "kitti":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=0.1)
            return [optimizer], [scheduler]
