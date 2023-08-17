import torch


def collate_fn(batch):
    data = {}
    imgs = []
    CP_mega_matrices = []
    targets = []
    frame_ids = []
    sequences = []

    cam_ks = []
    T_velo_2_cams = []
    frustums_masks = []
    frustums_class_dists = []

    scale_3ds = batch[0]["scale_3ds"]
    for scale_3d in scale_3ds:
        data["projected_pix_{}".format(scale_3d)] = []
        data["fov_mask_{}".format(scale_3d)] = []

    for idx, input_dict in enumerate(batch):
        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).double())
        T_velo_2_cams.append(torch.from_numpy(input_dict["T_velo_2_cam"]).float())

        if "frustums_masks" in input_dict:
            frustums_masks.append(torch.from_numpy(input_dict["frustums_masks"]))
            frustums_class_dists.append(
                torch.from_numpy(input_dict["frustums_class_dists"]).float()
            )

        for key in data:
            data[key].append(torch.from_numpy(input_dict[key]))

        img = input_dict["img"]
        imgs.append(img)

        frame_ids.append(input_dict["frame_id"])
        sequences.append(input_dict["sequence"])
        
        
        target = torch.from_numpy(input_dict["target"])
        targets.append(target)
        CP_mega_matrices.append(input_dict["CP_mega_matrix"])            

    ret_data = {
        "frame_id": frame_ids,
        "sequence": sequences,
        "frustums_class_dists": frustums_class_dists,
        "frustums_masks": frustums_masks,
        "cam_k": torch.stack(cam_ks),
        "T_velo_2_cam": torch.stack(T_velo_2_cams),
        "img": torch.stack(imgs),
        "CP_mega_matrices": torch.stack(CP_mega_matrices),
        "target": torch.stack(targets)
    }
    for key in data:
        ret_data[key] = data[key]
    ret_data["projected_pix_2"] = torch.stack(ret_data["projected_pix_2"]).to(torch.float32)
    ret_data["fov_mask_2"] = torch.stack(ret_data["fov_mask_2"]).to(torch.bool)
    return ret_data
