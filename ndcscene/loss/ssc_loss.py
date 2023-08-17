import torch
import torch.nn as nn
import torch.nn.functional as F


def KL_sep(p, target, reduction='sum'):
    """
    KL divergence
    """
    kl_term = F.kl_div(torch.log(p), target, reduction=reduction)
    return kl_term

def frustum_loss(pred, frustums_masks, frustums_class_dists):
    b, f, h, w, d = frustums_masks.shape
    c = pred.shape[1]
    pred_prob = F.softmax(pred, dim=1) # [b, c, H, W, D]
    pred_prob = torch.einsum('bfhwd,bchwd->bfc', frustums_masks.float(), pred_prob).reshape(b*f, c)
    frustums_class_dists = frustums_class_dists.reshape(b*f, c) # [b*f, c]
    nonzero_mask = (pred_prob > 0).any(1) & (frustums_class_dists > 0).any(1)
    pred_prob = pred_prob[nonzero_mask]
    frustums_class_dists = frustums_class_dists[nonzero_mask]
    pred_prob = pred_prob / pred_prob.sum(1, keepdim=True) # [b*f, c]
    frustums_class_dists = frustums_class_dists / frustums_class_dists.sum(1, keepdim=True) # [b*f, c]
    frustum_loss = KL_sep(pred_prob, frustums_class_dists, reduction='batchmean')

    return frustum_loss

def geo_scal_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :].reshape(-1)
    nonempty_probs = 1 - empty_probs
    ssc_target = ssc_target.reshape(-1)

    # Remove unknown voxels
    mask = ssc_target != 255
    target = ssc_target[mask]
    empty_probs = empty_probs[mask]
    nonempty_probs = nonempty_probs[mask]

    nonempty_target = target != 0

    intersection = nonempty_probs[nonempty_target].sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = empty_probs[~nonempty_target].sum() / (~nonempty_target).sum()
    return -precision.log() - recall.log() - spec.log()


def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    bs, nc, h, w, d = pred.shape
    pred = F.softmax(pred, dim=1).permute(0, 2, 3, 4, 1).reshape(-1, nc)
    ssc_target = ssc_target.reshape(-1)
    mask = (ssc_target != 255)
    pred = pred[mask]
    ssc_target = ssc_target[mask]

    loss = 0
    count = 0
    for i in range(0, nc):
        # Get probability of class i
        p = pred[:, i]
        completion_target = (ssc_target == i)
        if completion_target.any():
            count += 1.0
            nominator = p[completion_target].sum()
            loss_class = 0
            if (p > 0).any():
                precision = nominator / p.sum()
                loss_precision = -precision.log()
                loss_class += loss_precision
            if completion_target.any():
                recall = nominator / completion_target.sum()
                loss_recall = -recall.log()
                loss_class += loss_recall
            if (~completion_target).any():
                specificity = (1 - p)[~completion_target].sum() / (~completion_target).sum()
                loss_specificity = -specificity.log()
                loss_class += loss_specificity
            loss += loss_class
    return loss / count


def CE_ssc_loss(pred, target, class_weights):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="mean"
    )
    loss = criterion(pred, target.long())

    return loss

def miou_loss(pred, ssc_target, smooth=1):
    prob = pred.softmax(dim=1) # [b, c, h, w, d]
    ssc_target = F.one_hot(ssc_target.long(), num_classes=prob.shape[1]).permute(0, 4, 1, 2, 3) # [b, c, h, w, d]
    inter = prob * ssc_target
    union = prob + ssc_target - inter
    return -((inter.sum() + smooth) / (union.sum() + smooth)).log()