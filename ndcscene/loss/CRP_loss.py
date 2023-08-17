import torch


# def compute_super_CP_multilabel_loss(pred_logits, CP_mega_matrices):
#     logits = []
#     labels = []
#     bs, n_relations, _, _ = pred_logits.shape
#     for i in range(bs):
#         pred_logit = pred_logits[i, :, :, :].permute(
#             0, 2, 1
#         )  # n_relations, N, n_mega_voxels
#         CP_mega_matrix = CP_mega_matrices[i]  # n_relations, N, n_mega_voxels
#         logits.append(pred_logit.reshape(n_relations, -1))
#         labels.append(CP_mega_matrix.reshape(n_relations, -1))

#     logits = torch.cat(logits, dim=1).T  # M, 4
#     labels = torch.cat(labels, dim=1).T  # M, 4

#     cnt_neg = (labels == 0).sum(0)
#     cnt_pos = labels.sum(0)
#     pos_weight = cnt_neg / cnt_pos
#     criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#     loss_bce = criterion(logits, labels.float())
#     return loss_bce

# def compute_super_CP_multilabel_loss(pred_logits, CP_mega_matrices):
#     # logits = []
#     # labels = []
#     # _, n_relations, _, _ = pred_logits.shape
#     # for i in range(bs):
#     #     pred_logit = pred_logits[i, :, :, :].permute(
#     #         0, 2, 1
#     #     )  # n_relations, N, n_mega_voxels
#     #     CP_mega_matrix = CP_mega_matrices[i]  # n_relations, N, n_mega_voxels
#     #     logits.append(pred_logit.reshape(n_relations, -1))
#     #     labels.append(CP_mega_matrix.reshape(n_relations, -1))

#     # logits = torch.cat(logits, dim=1).T  # M, 4
#     # labels = torch.cat(labels, dim=1).T  # M, 4

#     n_relations = pred_logits.shape[1]
#     logits = pred_logits.permute(0, 3, 2, 1).reshape(-1, n_relations)
#     labels = CP_mega_matrices.permute(0, 2, 3, 1).reshape(-1, n_relations)

#     cnt_neg = (labels == 0).sum(0)
#     cnt_pos = labels.sum(0)
#     pos_weight = cnt_neg / cnt_pos
#     criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#     loss_bce = criterion(logits, labels.float())
#     return loss_bce

def compute_super_CP_multilabel_loss(pred_logits, CP_mega_matrices):
    CP_mega_matrices, mask = CP_mega_matrices[:, :, :, :-1].long(), CP_mega_matrices[:, :, :, -1].bool()
    n_relations = CP_mega_matrices.shape[3]
    valid_mask = mask.reshape(-1)
    logits = pred_logits.reshape(-1, n_relations)[valid_mask]
    labels = CP_mega_matrices.reshape(-1, n_relations)[valid_mask]

    cnt_neg = (labels == 0).sum()
    cnt_pos = labels.sum()
    pos_weight = cnt_neg / cnt_pos
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_bce = criterion(logits, labels.float())
    return loss_bce