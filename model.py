
from ast import Not
import torch
from torch import nn


from utils.pointnet import PointNetEncoder_0
from utils.transformer import TransformerDecoderLayer, TransformerDecoder, _get_clones
from config import DENOISE, ABLATION
from utils.dataloader_EC import distance_point2edge_tensor

class PointDenoising(nn.Module):
    def __init__(self):
        super(PointDenoising, self).__init__()

        self.dim = DENOISE.dim
        self.pts_encoder = PointNetEncoder_0(out_channel=self.dim)

        # self.positional_encoding = nn.Linear(3, self.dim)

        self.extra_noise = nn.Parameter(torch.FloatTensor(DENOISE.mini_point, self.dim)) # [N, C]

        self.decoder_norm = nn.LayerNorm(self.dim)
        decoder_layer = TransformerDecoderLayer(self.dim, DENOISE.nhead, 
                                        self.dim*4, DENOISE.dropout, activation="relu", 
                                        norm=None, normalize_before=True,
                                        remove_self_attn=False, 
                                        add_self_attn=False,
                                        bilateral_attention = False
                                        )

        self.decoder = _get_clones(decoder_layer, DENOISE.num_decoder)

        ### coordinate regression
        self.coord_layer = nn.Sequential(
            nn.Conv1d(self.dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

        if ABLATION.edge_constraint:
            self.dist_layer = nn.Sequential(
                nn.Conv1d(self.dim, 64, 1),
                nn.ReLU(),
                nn.Conv1d(64, 1, 1)
            )
            self.add_weight = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))
            self.register_parameter("add_weight", self.add_weight)
        
        self.init_weight()
        self.MSELoss = nn.MSELoss(reduction='mean')
        self.L1Loss = nn.L1Loss(reduction='mean')

    def init_weight(self):
        # 参数初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                torch.nn.init.ones_(m.weight)


        nn.init.normal_(self.extra_noise, 0, 0.1)
    
    def forward(self, xyz, gt=None, edges=None):
        # xyz: [B, 3, N]
        # gt: [B, 3, N]
        # edge : [B, 6, M]

        B, _, N = xyz.shape
        '''
        如果直接添加噪声到点云中，-->  同时该模式下remove_query_pos必须为True, 噪声维度为3

        '''

        feature = self.pts_encoder(xyz).permute(0, 2, 1) # [B, C, N] --> [B, N, C]
        pos = None

        query_embed = self.extra_noise.unsqueeze(0).repeat(B, 1, 1)  # [B, N, C]
        tgt = feature
        
        features = []
        for decoder_layer in self.decoder:
            output = decoder_layer(tgt=tgt, memory=feature, pos=pos, query_pos=query_embed+feature)
            features.append(self.decoder_norm(output))
            tgt = output

        features = torch.stack(features)    # [s, B, N, C]
        feature = features[-1]      # [B, N, C]

        coord = self.coord_layer(feature.permute(0, 2, 1))   # [B, 3, N]
        
        if gt is None:
            return coord
        else:
            mse_loss = self.MSELoss(coord, gt)
            L1_loss = self.L1Loss(coord, gt)

            loss_log = {
                'denoise_mse': mse_loss.cpu().detach().item(),
                'denoise_l1': L1_loss.cpu().detach().item()
            }
        
            if ABLATION.edge_constraint and edges is not None and edges.shape[1]==6:

                # 首先计算网络预测的点到边缘之间的距离
                pred_dist = self.dist_layer(feature.permute(0, 2, 1)).squeeze(1) # [B, 1, N] --> [B, N]
                pred_dist = torch.clamp(pred_dist, min=0, max=0.5)    # [B, N]
                # 然后计算预测点与边缘之间的距离
                coord2edge, _ = distance_point2edge_tensor(coord.permute(0,2,1), edges.permute(0,2,1))     # [B, N]
                coord2edge = torch.clamp(coord2edge, max=0.5)    # [B, N]

                # 计算边缘距离回归损失：预测距离与预测点2边缘之间的距离越小越好
                dist_mseloss = self.MSELoss(coord2edge, pred_dist)
                # 根据预测距离的大小判断哪些点为边缘点
                # weight = F.relu(0.5-self.add_weight/20000.0)
                weight = torch.max(0.5-self.add_weight/20000.0, torch.zeros_like(self.add_weight))
                rough_dist = weight * coord2edge + (1-weight) * pred_dist  # 将两个距离加权
                # 计算边缘损失：边缘点到边缘的距离越小越好    
                edge_mask = torch.where(rough_dist<DENOISE.edge_thres, torch.ones_like(rough_dist), torch.zeros_like(rough_dist))
                edge_dist = edge_mask * coord2edge
                edge_loss = torch.sum(edge_dist) / (torch.sum(edge_mask) + 1)
                

                loss_log["denoise_dist"] = dist_mseloss.cpu().detach().item() 
                loss_log["denoise_edge"] = edge_loss.cpu().detach().item() 

            loss =  L1_loss
           
            if ABLATION.edge_constraint and edges is not None and edges.shape[1]==6:
                loss = loss + DENOISE.edge_loss_weight * (dist_mseloss + 10 * edge_loss)
            
            return loss, loss_log, coord, feature