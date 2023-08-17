import math
import torch
import torch.nn as nn
from ndcscene.models.modules import (
    Process,
    ASPP,
)


# class CPMegaVoxels(nn.Module):
#     def __init__(self, feature, size, n_relations=4, bn_momentum=0.0003):
#         super().__init__()
#         self.size = size
#         self.n_relations = n_relations
#         print("n_relations", self.n_relations)
#         self.flatten_size = size[0] * size[1] * size[2]
#         self.feature = feature
#         self.context_feature = feature * 2
#         self.flatten_context_size = (size[0] // 2) * (size[1] // 2) * (size[2] // 2)
#         padding = ((size[0] + 1) % 2, (size[1] + 1) % 2, (size[2] + 1) % 2)
        
#         self.mega_context = nn.Sequential(
#             nn.Conv3d(
#                 feature, self.context_feature, stride=2, padding=padding, kernel_size=3
#             ),
#         )
#         self.flatten_context_size = (size[0] // 2) * (size[1] // 2) * (size[2] // 2)

#         self.context_prior_logits = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Conv3d(
#                         self.feature,
#                         self.flatten_context_size,
#                         padding=0,
#                         kernel_size=1,
#                     ),
#                 )
#                 for i in range(n_relations)
#             ]
#         )
#         self.aspp = ASPP(feature, [1, 2, 3])

#         self.resize = nn.Sequential(
#             nn.Conv3d(
#                 self.context_feature * self.n_relations + feature,
#                 feature,
#                 kernel_size=1,
#                 padding=0,
#                 bias=False,
#             ),
#             Process(feature, nn.BatchNorm3d, bn_momentum, dilations=[1]),
#         )

#     def forward(self, input):
#         ret = {}
#         bs = input.shape[0]

#         x_agg = self.aspp(input)

#         # get the mega context
#         x_mega_context_raw = self.mega_context(x_agg) # [b, 2*c, h/2, w/2, d/2]
#         x_mega_context = x_mega_context_raw.reshape(bs, self.context_feature, -1) # [b, 2*c, h/2 * w/2 * d/2]
#         x_mega_context = x_mega_context.permute(0, 2, 1) # [b, h/2 * w/2 * d/2, 2*c]

#         # get context prior map
#         x_context_prior_logits = []
#         x_context_rels = []
#         for rel in range(self.n_relations):

#             # Compute the relation matrices
#             x_context_prior_logit = self.context_prior_logits[rel](x_agg) # [b, h/2 * w/2 * d/2, h, w, d]
#             x_context_prior_logit = x_context_prior_logit.reshape(
#                 bs, self.flatten_context_size, self.flatten_size
#             ) # [b, h/2 * w/2 * d/2, h * w * d]
#             x_context_prior_logits.append(x_context_prior_logit.unsqueeze(1))

#             x_context_prior_logit = x_context_prior_logit.permute(0, 2, 1) # [b, h * w * d, h/2 * w/2 * d/2]
#             x_context_prior = torch.sigmoid(x_context_prior_logit) # [b, h * w * d, h/2 * w/2 * d/2]

#             # Multiply the relation matrices with the mega context to gather context features
#             x_context_rel = torch.bmm(x_context_prior, x_mega_context) # [b, h * w * d, 2*c]
#             x_context_rels.append(x_context_rel)

#         x_context = torch.cat(x_context_rels, dim=2) # [b, h * w * d, 8*c]
#         x_context = x_context.permute(0, 2, 1) # [b, 8*c, h * w * d]
#         x_context = x_context.reshape(
#             bs, x_context.shape[1], self.size[0], self.size[1], self.size[2]
#         ) # [b, 8*c, h, w, d]

#         x = torch.cat([input, x_context], dim=1) # [b, 9*c, h, w, d]
#         x = self.resize(x) # [b, c, h, w, d]

#         x_context_prior_logits = torch.cat(x_context_prior_logits, dim=1) # [b, 4, h/2 * w/2 * d/2, h * w * d]
#         ret["P_logits"] = x_context_prior_logits # [b, 4, h/2 * w/2 * d/2, h * w * d]
#         ret["x"] = x # [b, c, h, w, d]

#         return ret

# class CPMegaVoxels(nn.Module):
#     def __init__(self, feature, size, n_relations=4, bn_momentum=0.0003):
#         super().__init__()
#         self.size = size
#         self.n_relations = n_relations
#         print("n_relations", self.n_relations)
#         self.flatten_size = size[0] * size[1] * size[2]
#         self.feature = feature
#         self.flatten_context_size = (size[0] // 2) * (size[1] // 2) * (size[2] // 2)
#         padding = ((size[0] + 1) % 2, (size[1] + 1) % 2, (size[2] + 1) % 2)
        
#         self.aspp = ASPP(feature, [1, 2, 3])

#         self.q_1 = nn.Conv3d(feature, n_relations * 64, kernel_size=1, stride=1, padding=0)
#         self.k_1 = nn.Conv3d(feature, n_relations * 64, kernel_size=3, stride=2, padding=padding)
#         self.q_2 = nn.Conv3d(feature, 200, kernel_size=1, stride=1, padding=0)
#         self.k_2 = nn.Conv3d(feature, 200, kernel_size=3, stride=2, padding=padding)
#         self.v_2 = nn.Conv3d(feature, n_relations * 2 * feature, kernel_size=3, stride=2, padding=padding)

#         self.resize = nn.Sequential(
#             nn.Conv3d(
#                 n_relations * 2 * feature + feature,
#                 feature,
#                 kernel_size=1,
#                 padding=0,
#                 bias=True,
#             ),
#             nn.Conv3d(
#                 feature,
#                 feature // 4,
#                 kernel_size=3,
#                 padding=1,
#                 bias=True,
#             ),
#             nn.Conv3d(
#                 feature // 4,
#                 feature,
#                 kernel_size=1,
#                 padding=0,
#                 bias=True,
#             ),
#             nn.BatchNorm3d(feature, momentum=0.001),
#         )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, input):
#         ret = {}
#         b, n = input.shape[0], self.n_relations

#         x_agg = self.aspp(input)

#         q_1 = self.q_1(x_agg).reshape(b * n, -1, self.flatten_size) # [b * n, c, h * w * d]
#         k_1 = self.k_1(x_agg).reshape(b * n, -1, self.flatten_context_size) # [b * n, c, h / 2 * w / 2 * d / 2]
#         k_1 = k_1.div_(math.sqrt(k_1.shape[1]))
#         ret["P_logits"] = torch.bmm(q_1.permute(0, 2, 1), k_1).reshape(b, n, self.flatten_size, self.flatten_context_size)

#         q_2 = self.q_2(x_agg).reshape(b * n, -1, self.flatten_size) # [b * 4, c, h * w * d]
#         k_2 = self.k_2(x_agg).reshape(b * n, -1, self.flatten_context_size) # [b * 4, c, h / 2 * w / 2 * d / 2]
#         k_2 = k_2.div_(math.sqrt(k_2.shape[1]))
#         v_2 = self.v_2(x_agg).reshape(b * n, -1, self.flatten_context_size) # [b * 4, c, h / 2 * w / 2 * d / 2]
#         a_2 = torch.bmm(k_2.permute(0, 2, 1), q_2).reshape(b * 4, self.flatten_context_size, self.flatten_size).softmax(dim=1) # [b * 4, h / 2 * w / 2 * d / 2, h * w * d]
#         x = torch.bmm(v_2, a_2).reshape(b, -1, *input.shape[2:])
#         # ret["x"] = self.relu(self.resize(torch.cat([input, x], dim=1)) + input)
#         ret["x"] = self.relu(self.resize(torch.cat([input, x], dim=1)))

#         return ret

# class CPMegaVoxels(nn.Module):
#     def __init__(self, feature, size, n_relations=4, bn_momentum=0.0003):
#         super().__init__()
#         self.size = size
#         self.n_relations = n_relations
#         print("n_relations", self.n_relations)
#         self.flatten_size = size[0] * size[1] * size[2]
#         self.feature = feature
#         self.flatten_context_size = (size[0] // 2) * (size[1] // 2) * (size[2] // 2)
#         padding = ((size[0] + 1) % 2, (size[1] + 1) % 2, (size[2] + 1) % 2)
        
#         self.aspp = ASPP(feature, [1, 2, 3])

#         self.p = nn.Conv3d(feature, n_relations * 9 * 9 * 9, kernel_size=1, stride=1, padding=0)
#         self.q = nn.Conv3d(feature, 200, kernel_size=1, stride=1, padding=0)
#         self.k = nn.Conv3d(feature, 200, kernel_size=3, stride=2, padding=padding)
#         self.v = nn.Conv3d(feature, n_relations * 2 * feature, kernel_size=3, stride=2, padding=padding)

#         self.resize = nn.Sequential(
#             nn.Conv3d(
#                 n_relations * 2 * feature + feature,
#                 feature,
#                 kernel_size=1,
#                 padding=0,
#                 bias=True,
#             ),
#             nn.Conv3d(
#                 feature,
#                 feature // 4,
#                 kernel_size=3,
#                 padding=1,
#                 bias=True,
#             ),
#             nn.Conv3d(
#                 feature // 4,
#                 feature,
#                 kernel_size=1,
#                 padding=0,
#                 bias=True,
#             ),
#             nn.BatchNorm3d(feature, momentum=0.001),
#         )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, input):
#         ret = {}
#         b, n = input.shape[0], self.n_relations

#         x_agg = self.aspp(input)

#         ret["P_logits"] = self.p(x_agg).reshape(b, n, 9 * 9 * 9, self.flatten_size).permute(0, 2, 3, 1) # [b, 9 * 9 * 9, h * w * d, 4]

#         q = self.q(x_agg).reshape(b * n, -1, self.flatten_size) # [b * 4, c, h * w * d]
#         k = self.k(x_agg).reshape(b * n, -1, self.flatten_context_size) # [b * 4, c, h / 2 * w / 2 * d / 2]
#         k = k.div_(math.sqrt(k.shape[1]))
#         v = self.v(x_agg).reshape(b * n, -1, self.flatten_context_size) # [b * 4, c, h / 2 * w / 2 * d / 2]
#         a = torch.bmm(k.permute(0, 2, 1), q).reshape(b * 4, self.flatten_context_size, self.flatten_size).softmax(dim=1) # [b * 4, h / 2 * w / 2 * d / 2, h * w * d]
#         x = torch.bmm(v, a).reshape(b, -1, *input.shape[2:])
#         ret["x"] = self.relu(self.resize(torch.cat([input, x], dim=1)))

#         return ret

class CPMegaVoxels(nn.Module):
    def __init__(self, feature, size, n_relations=4, bn_momentum=0.0003, stride_last=1):
        super().__init__()
        self.n_relations = n_relations
        self.flatten_size = size[0] * size[1] * size[2]
        
        self.aspp = ASPP(feature, [1, 2, 3])
        self.p = nn.Conv3d(feature, n_relations * 9 * 9 * 9, kernel_size=1, stride=(1, 1, stride_last), padding=0)
        self.resize = nn.Sequential(
            nn.Conv3d(
                feature + feature,
                feature // 2,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
            nn.Conv3d(
                feature // 2,
                feature // 2,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.Conv3d(
                feature // 2,
                feature,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm3d(feature, momentum=0.001),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        ret = {}
        b, n = input.shape[0], self.n_relations

        x_agg = self.aspp(input)

        ret["P_logits"] = self.p(x_agg).reshape(b, n, 9 * 9 * 9, self.flatten_size).permute(0, 2, 3, 1) # [b, 9 * 9 * 9, h * w * d, 4]

        ret["x"] = self.relu(self.resize(torch.cat([input, x_agg], dim=1)) + x_agg)

        return ret