import torch.nn as nn
import torch
import torch.nn.functional as F
from pyexpat import features

from models.risurconv_utils import RISurConvSetAbstraction

class get_model(nn.Module):
    def __init__(self, num_class, n=1, normal_channel=False):
        super(get_model, self).__init__()
        self.normal_channel = normal_channel
        # 输入通道逻辑：
        # - normal_channel=False: [x,y,z, potential, normal_potential] -> 5通道
        # - normal_channel=True:  [x,y,z, potential, normal_potential, nx,ny,nz] -> 8通道
        in_channel = 5 if not normal_channel else 8

        # print('in_channel - 3:',str(in_channel - 3))
        # 修改各层in_channel（仅第一层需要，后续层由前一层的out_channel决定）
        self.sc0 = RISurConvSetAbstraction(
            npoint=512 * n, radius=0.12, nsample=8,
            in_channel=in_channel - 3,  # 输入特征通道数（减去坐标3维）
            out_channel=32,
            group_all=False
        )
        # 其他层保持不变...
        self.sc1 = RISurConvSetAbstraction(npoint=256 * n, radius=0.16, nsample=16, in_channel=32, out_channel=64,
                                           group_all=False)
        self.sc2 = RISurConvSetAbstraction(npoint=128 * n, radius=0.24, nsample=32, in_channel=64, out_channel=128,
                                           group_all=False)
        self.sc3 = RISurConvSetAbstraction(npoint=64 * n, radius=0.48, nsample=64, in_channel=128, out_channel=256,
                                           group_all=False)
        self.sc4 = RISurConvSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, out_channel=512,
                                           group_all=True)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout=0.05)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, xyz):
        if xyz.dim() == 3:  # 如果是3D张量 [B, N, C]
            B, N, _ = xyz.shape
        elif xyz.dim() == 2:  # 已经是2D
            B, _ = xyz.shape
        else:
            raise ValueError(f"Unexpected input shape: {xyz.shape}")
        if self.normal_channel:
            norm = xyz[:, :, 5:8]
            features = xyz[:,:,3:5]
            xyz = xyz[:, :, :3]
        else:
            # compute the LRA and use as normal
            norm = None
            print(xyz.shape)
            features = xyz[:,:,3:5]
            xyz = xyz[:,:,:3]

        l0_xyz, l0_norm, l0_points = self.sc0(xyz, norm,features)
        l1_xyz, l1_norm, l1_points = self.sc1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sc2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sc3(l2_xyz, l2_norm, l2_points)
        l4_xyz, l4_norm, l4_points = self.sc4(l3_xyz, l3_norm, l3_points)

        x = l4_points.permute(0, 2, 1)
        x = self.transformer_encoder(x)
        globle_x = x.view(B, 512)
        # globle_x = torch.max(l4_feature, 2)[0]

        x = self.drop1(F.relu(self.bn1(self.fc1(globle_x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l4_points

class get_loss(nn.Module):
    def __init__(self):
            super(get_loss, self).__init__()

    def forward(self, pred, target):
            if isinstance(pred, tuple):  # 如果 pred 是元组，只取分类 logits
                pred = pred[0]
            total_loss = F.nll_loss(pred, target)
            return total_loss
