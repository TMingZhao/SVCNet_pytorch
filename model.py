
import torch, math
from torch import nn
import torch.nn.functional as F

from utils.pointnet import PointNetEncoder_0
from config import DENOISE, ABLATION
from utils.dataloader_EC import distance_point2edge_tensor


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "LeakyReLU":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def attention(q, k, v, d_k, dropout=None):
    # [2, 4, 8, 64]     # [2, 1, 8, 256]
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k) # [2, 4, 8, 8]

    scores = F.softmax(scores, dim=-1)  # [2, 4, 8, 8]
    
    if dropout is not None:
        scores = dropout(scores)    # [2, 4, 8, 8]
        
    output = torch.matmul(scores, v)    # [2, 4, 8, 64]
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

       
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value):
        # [2, 8, 256]

        bs = query.size(0)
        
    
        # perform linear operation and split into N heads
        k = self.k_linear(key).contiguous().view(bs, -1, self.h, self.d_k) # [2, 8, 4, 64]
        q = self.q_linear(query).contiguous().view(bs, -1, self.h, self.d_k)
        v = self.v_linear(value).contiguous().view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)    # [2, 4, 8, 64]
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output, scores
    

class CommonalityCaptureLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()


        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)


    def forward(self, tgt, memory):

        tgt2 = self.norm1(tgt)
        q = tgt
        k = memory
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
            

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=tgt2,
                                   key=memory,
                                   value=memory)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt


class PointDenoising(nn.Module):
    def __init__(self):
        super(PointDenoising, self).__init__()

        self.dim = DENOISE.dim
        self.pts_encoder = PointNetEncoder_0(out_channel=self.dim)

        self.extra_noise = nn.Parameter(torch.FloatTensor(DENOISE.mini_point, self.dim)) # [N, C]

        self.norm = nn.LayerNorm(self.dim)
        self.common_capture = CommonalityCaptureLayer(self.dim, DENOISE.nhead, 
                                        self.dim*4, DENOISE.dropout, activation="relu")

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

        feature = self.pts_encoder(xyz).permute(0, 2, 1) # [B, C, N] --> [B, N, C]

        noise_vector = self.extra_noise.unsqueeze(0).repeat(B, 1, 1)  # [B, N, C]

        self_variation = feature + noise_vector
        

        feature = self.common_capture(tgt=self_variation, memory=feature)
        feature = self.norm(feature)

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

