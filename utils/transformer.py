"""
Originally from : https://github.com/facebookresearch/detr/blob/master/models/transformer.py
Modify by Tianming Zhao
Date: August 2019
"""

import sys
import os
sys.path.append(os.getcwd())
import copy
import imp
import re
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math

from torch.nn.parameter import Parameter
from torch.nn.functional import linear


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


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def attention(q, k, v, d_k, corr=None, dropout=None):
    # [2, 4, 8, 64]     # [2, 1, 8, 256]
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k) # [2, 4, 8, 8]
    
    if corr is not None:
        corr = corr.unsqueeze(1)   # [2, 1, 8, 8 ]
        scores = scores + corr

    scores = F.softmax(scores, dim=-1)  # [2, 4, 8, 8]
    
    if dropout is not None:
        scores = dropout(scores)    # [2, 4, 8, 8]
        
    output = torch.matmul(scores, v)    # [2, 4, 8, 64]
    return output

# q^T * k * V + K^T * q * q
def attention_cross(q, k, v, d_k, corr=None, dropout=None):
    # [2, 4, 8, 64]     # [2, 1, 8, 256]
    scores_1 = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k) # [2, 4, 8, 8]
    scores_2 = scores_1.transpose(-2, -1)
    if corr is not None:
        corr_1 = corr.unsqueeze(1)   # [2, 1, 8, 8 ]
        corr_2 = corr_1.transpose(-2, -1)
        scores_1 = scores_1 + corr_1
        scores_2 = scores_2 + corr_2

    scores_1 = F.softmax(scores_1, dim=-1)  # [2, 4, 8, 8]
    scores_2 = F.softmax(scores_2, dim=-1)  # [2, 4, 8, 8]
    
    if dropout is not None:
        scores_1 = dropout(scores_1)    # [2, 4, 8, 8]
        scores_2 = dropout(scores_2)    # [2, 4, 8, 8]
        
    output_1 = torch.matmul(scores_1, v)    # [2, 4, 8, 64]
    output_2 = torch.matmul(scores_2, q)    # [2, 4, 8, 64]

    outputs = torch.stack([output_1, output_2])  # [2, 2, 4, 8, 64]
    output = torch.max(outputs, dim=0)[0]  # [2, 4, 8, 64]

        
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1, bilateral_attention=False):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        # self.in_proj_weight = Parameter(torch.empty(3 * d_model, d_model))
        # self.in_proj_bias = Parameter(torch.empty(3 * d_model))

        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, corr=None, key_padding_mask=None):
        # [2, 8, 256]

        bs = query.size(0)
        
        
        # q = linear(query, self.in_proj_weight[0:self.d_model, :], self.in_proj_bias[0:self.d_model]).view(bs, -1, self.h, self.d_k) # [2, 8, 4, 64]
        # k = linear(key,   self.in_proj_weight[self.d_model:2*self.d_model, :], self.in_proj_bias[self.d_model:2*self.d_model]).view(bs, -1, self.h, self.d_k) # [2, 8, 4, 64]
        # v = linear(value, self.in_proj_weight[2*self.d_model:, :], self.in_proj_bias[2*self.d_model]).view(bs, -1, self.h, self.d_k) # [2, 8, 4, 64]

        # perform linear operation and split into N heads
        k = self.k_linear(key).contiguous().view(bs, -1, self.h, self.d_k) # [2, 8, 4, 64]
        q = self.q_linear(query).contiguous().view(bs, -1, self.h, self.d_k)
        v = self.v_linear(value).contiguous().view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)    # [2, 4, 8, 64]
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next


        scores = attention(q, k, v, self.d_k, corr, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output, scores

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", norm=None, normalize_before=False,
                 remove_self_attn=False, add_self_attn=False, bilateral_attention=False):
        super().__init__()
        self.remove_attn = remove_self_attn
        self.add_attn = add_self_attn
        if self.remove_attn:
            pass  
        else:
            self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        if self.add_attn:
            self.self_attn_1 = MultiHeadAttention(d_model, nhead, dropout=dropout)
            self.norm1_1 = nn.LayerNorm(d_model)
            self.dropout1_1 = nn.Dropout(dropout)

        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, bilateral_attention=bilateral_attention)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.norm = norm
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        if self.remove_attn:
            pass
        else:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        if self.add_attn:
            q = k = self.with_pos_embed(memory, pos)
            memory2 = self.self_attn_1(q, k, value=memory, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]
            memory = memory + self.dropout1_1(memory2)
            memory = self.norm1_1(memory)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        if self.remove_attn:
            pass 
        else:
            tgt2 = self.norm1(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt2, corr=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)

        if self.add_attn:
            memory2 = self.norm1_1(memory)
            q = k = self.with_pos_embed(memory2, pos)
            memory2 = self.self_attn_1(q, k, value=memory2, corr=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]
            memory = memory + self.dropout1_1(memory2)
            

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, corr=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if self.norm is not None:
            tgt = self.norm(tgt)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)




class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,   # [N, B, C]
                     corr_matrix: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)   # [N, B, C]
        src2 = self.self_attn(q, k, value=src, corr=corr_matrix,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    corr_matrix: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, corr=corr_matrix,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                corr_matrix: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, corr_matrix, src_key_padding_mask, pos)
        return self.forward_post(src, corr_matrix, src_key_padding_mask, pos)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                if self.norm is not None:
                    intermediate.append(self.norm(output))
                else:
                    intermediate.append(output)
            # if self.norm is not None:
            #     output = self.norm(output)
            # if self.return_intermediate:
            #     intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output    


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src,
                corr: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        intermediate = []
        
        for layer in self.layers:
            output = layer(output, corr_matrix=corr,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            if self.return_intermediate:
                if self.norm is not None:
                    intermediate.append(self.norm(output))
                else:
                    intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)     


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, global_dim=512):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, return_intermediate_dec)


        # global feature extractor
        self.global_dim = global_dim
        self.trans_dimen = nn.Conv1d(d_model, self.global_dim, 1)    # B,N,16
        self.maxpool = nn.AdaptiveMaxPool1d(1) # 平均池化操作
        global_encoder_layer = TransformerEncoderLayer(self.global_dim, nhead, dim_feedforward=2048 if self.global_dim>512 else self.global_dim*4,
                                            dropout=dropout, activation=activation, normalize_before=normalize_before)
        global_encoder_norm = nn.LayerNorm(self.global_dim) if normalize_before else None
        self.global_encoder = TransformerEncoder(global_encoder_layer, num_encoder_layers, global_encoder_norm)
        self.fuse_conv = nn.Conv1d(self.global_dim+d_model, d_model, 1)
            
            

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, corr, pos_embed, global_pos=None, global_corr=None):
        # src : [B, C, N]
        # mask: [B, N, N]
        # global_pos : [B, C, 1]


        bs, c, n = src.shape
        src = src.permute(0, 2, 1)      # [B, N, C]
        if pos_embed is not  None:
            pos_embed = pos_embed.permute(0, 2, 1) 


        # loacl feature
        memorys = self.encoder(src, corr=corr, pos=pos_embed)    # [num_layers, B, N, C]
        memory = memorys[-1]
        # global feature
        up_memory = self.trans_dimen(src.permute(0, 2, 1))   # [B, C, N]
        global_feature = self.maxpool(up_memory)   # [B, C, 1]
        global_feature = global_feature.permute(2, 0, 1)    # [1, B, C]
        if global_pos is not None:
            global_pos = global_pos.permute(2, 0, 1)   # [B, C, 1] --> [1, B, C]
        global_feature = self.global_encoder(global_feature, corr=global_corr, pos=global_pos)[-1]   # [1, B, C]
        global_feature = global_feature.permute(1, 0, 2).repeat(1, n, 1)    # [B, N, C]
        # feature fusion
        memory = self.fuse_conv(torch.cat((memory, global_feature), dim=2).permute(0, 2, 1)).permute(0, 2, 1)

        
        return memory.permute(0, 2, 1)
