import os
import argparse
import torch
import numpy as np

from utils.pc_utils import read_ply, save_ply
from utils import pc_utils
from utils.point_util import farthest_point_sample, index_points, knn_point, normalize_point_batch
from tqdm import tqdm
from torch.autograd import Variable
from utils import operations
from tqdm import tqdm
import config
from config import DENOISE
from model import PointDenoising

import importlib
P3D_found = importlib.util.find_spec("pytorch3d") is not None
if P3D_found:
    import pytorch3d.ops

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def get_correction_points(xyz, pred_xyz, n_neighbours=100):
    # xyz, pred_xyz: torch.tensor[B, N, 3]
    # n_neighbours: number


    # Chamfer = chamfer_3DDist()
    # dist1, dist2, idx1, idx2 = Chamfer(xyz, pred_xyz)
    # chamfer_dist = (torch.mean(dist1)) + (torch.mean(dist2))
    # xyz = index_points(xyz, idx1.long()) # [1, N, 3]
    # pred_xyz = index_points(pred_xyz, idx2.long()) # [1, N, 3]

    # 得到最近邻的索引
    idx = knn_point(n_neighbours, xyz, xyz) # [1, N, 100]   
    displacement_vectors = pred_xyz - xyz   # [1, N, 3]
    new_displacement = index_points(displacement_vectors, idx) # [1, N, 100, 3]
    new_displacement = new_displacement.mean(dim=2, keepdim=False)  # [1, N, 3]
    new_xyz = pred_xyz - new_displacement

   
    return new_xyz


def points2patches(xyz, npoint, nsample):
    # N, C = xyz.shape
    xyz = torch.from_numpy(xyz).unsqueeze(0)   # [1, N, 3]
    fps_idx = farthest_point_sample(xyz, npoint) # [1, npoint]
    new_xyz = index_points(xyz, fps_idx)    #[1, npoint, C]
    idx = knn_point(nsample, xyz, new_xyz)  # [1, npoint, nsample]
    grouped_xyz = index_points(xyz, idx) # [1, npoint, nsample, C]
    grouped_xyz = grouped_xyz.squeeze(0)    # [npoint, nsample, C]
    return grouped_xyz, idx

def pts_split(xyz, big_patch=10000):
    pts_list = []
    residual_pts = torch.from_numpy(xyz).unsqueeze(0)  # [1, N, 3]
    if not P3D_found:
        residual_pts = residual_pts.cuda()
    while(residual_pts.shape[1] >= big_patch*2):
        if not P3D_found:
            fps_idx, new_patch = operations.furthest_point_sample(residual_pts, big_patch, NCHW=False)   # [1, N] [1, N, 3]
        else:
            fps_idx = farthest_point_sample(residual_pts, big_patch) # [1, npoint]
            new_patch = index_points(residual_pts, fps_idx)    #[1, npoint, C]
        pts_list.append(new_patch[0].cpu().numpy())

        all_idx = np.arange(0, residual_pts.shape[1], dtype=np.int64)
        fps_idx = fps_idx.squeeze(0).cpu().numpy()
        res_idx = sorted(list(set(all_idx).difference(set(fps_idx))))
        res_idx = torch.tensor(res_idx).unsqueeze(0).to(residual_pts.device)    # [1, N-n, 3]
        if not P3D_found:
            residual_pts = operations.gather_points(residual_pts.transpose(2, 1).contiguous(), res_idx).transpose(2, 1).contiguous()
        else:
            residual_pts = index_points(residual_pts, res_idx)    #[1, npoint, C]


    pts_list.append(residual_pts[0].cpu().numpy())
    return pts_list
# 点云优化函数
def optimize_once(net, points, num_patch, device, batch_size=8):
    # input : [N, 3]
    # output : [rN, 3]

    # point cloud pretreat
    num_points = points.shape[0]
    points, all_centroid, all_radius = pc_utils.normalize_point_cloud(points)   # [N, 3] [1, 3] [1, 1]
    net.eval()

    num_patch = min(num_patch, num_points)
    a = int(num_points / num_patch) * config.repeatability
    patches, idx = points2patches(points, a, num_patch)
    
    up_patch_list = []

    use_tqdm = False
    if use_tqdm:
        pbar = tqdm(total=len(patches), desc='process')
        pbar.update(0)
    for i in range(0, len(patches), batch_size):
        
        
        i_s = i
        i_e = (i+batch_size) if ((i+batch_size) < len(patches)) else len(patches)
        patch = patches[i_s:i_e]

        if use_tqdm:
            pbar.update(len(patch))

        patch = patch.to(device)
        norm_patch, centroid, radius = normalize_point_batch(patch, False)# [B, N, 3]  [B, 1, 3]   [B, 1, 1]
        # patch_pos = torch.cat((centroid, radius), dim=2)   # [B, 1, 4]
        input_points = norm_patch.permute(0,2,1)
        # patch_pos = patch_pos.permute(0, 2, 1)   # [B, 4, 1]
        input_points = Variable(input_points.to(device), requires_grad=False)
        # patch_pos = Variable(patch_pos.to(device), requires_grad=False)


        with torch.no_grad():
            up_patch = net(input_points)

        up_ratio = up_patch.shape[2] // input_points.shape[2]
        up_patch = up_patch.permute(0,2,1)
        up_patch = up_patch * radius + centroid

        

        up_patch_list.extend(up_patch)
    if use_tqdm:
        pbar.close()
    up_points = torch.cat(up_patch_list, dim=0) # [936, 32, 3] --> [29952, 3]
    
    
    # downsample
    up_points = up_points.unsqueeze(0)   # [1, N, 3]
    if not P3D_found:
        down_idx, up_points = operations.furthest_point_sample(up_points, up_ratio*num_points, NCHW=False)   # [1, N] [1, N, 3]
        up_points = up_points.cpu()
        
    else:   # 第二种下采样方式
        # up_points, _ = pytorch3d.ops.sample_farthest_points(up_points, K=int(up_ratio*num_points))
        # up_points = up_points.cpu()

        down_idx = farthest_point_sample(up_points, up_ratio*num_points)
        up_points = index_points(up_points, down_idx)
        up_points = up_points.cpu()
    
    if 0:
        # 首先找到与预测点一一对应的原始点
        idx =  idx.reshape(1, -1, 1)   # [1, 29952, 1]
        new_idx = index_points(idx, down_idx.long()).squeeze(2)   # [1, N, 1] --> [1, N]
        points = torch.from_numpy(points).unsqueeze(0)  # [1, N, 3]
        old_points = index_points(points, new_idx.long())  # [1, N, 3]
        up_points = get_correction_points(old_points, up_points)


    up_points = up_points[0, ...]   # [rN, 3]
    up_points = up_points.numpy() * all_radius + all_centroid

    return up_points

# 点云上采样函数，将输入点云上采样两倍后输出
def upsample_once(up_ratio, net, points, num_patch, device, batch_size=8):
    # input : [N, 3]
    # output : [rN, 3]

    # point cloud pretreat
    num_points = points.shape[0]
    points, all_centroid, all_radius = pc_utils.normalize_point_cloud(points)   # [N, 3] [1, 3] [1, 1]
    net.eval()

    num_patch = min(num_patch, num_points)
    a = int(num_points / num_patch) * config.repeatability * up_ratio
    patches, idx = points2patches(points, a, num_patch)
    
    up_patch_list = []

    use_tqdm = False
    if use_tqdm:
        pbar = tqdm(total=len(patches), desc='process')
        pbar.update(0)
    for i in range(0, len(patches), batch_size):
        
        
        i_s = i
        i_e = (i+batch_size) if ((i+batch_size) < len(patches)) else len(patches)
        patch = patches[i_s:i_e]

        if use_tqdm:
            pbar.update(len(patch))

        patch = patch.to(device)
        norm_patch, centroid, radius = normalize_point_batch(patch, False)# [B, N, 3]  [B, 1, 3]   [B, 1, 1]
        # patch_pos = torch.cat((centroid, radius), dim=2)   # [B, 1, 4]
        input_points = norm_patch.permute(0,2,1)
        # patch_pos = patch_pos.permute(0, 2, 1)   # [B, 4, 1]
        input_points = Variable(input_points.to(device), requires_grad=False)
        # patch_pos = Variable(patch_pos.to(device), requires_grad=False)


        with torch.no_grad():
            up_patch = net(input_points)

        up_patch = up_patch.permute(0,2,1)
        up_patch = up_patch * radius + centroid

        up_patch_list.extend(up_patch)
    if use_tqdm:
        pbar.close()
    up_points = torch.cat(up_patch_list, dim=0) # [936, 32, 3] --> [29952, 3]
    
    
    # downsample
    up_points = up_points.unsqueeze(0)   # [1, N, 3]
    if not P3D_found:
        down_idx, up_points = operations.furthest_point_sample(up_points, up_ratio*num_points, NCHW=False)   # [1, N] [1, N, 3]
        up_points = up_points.cpu()
        
    else:   # 第二种下采样方式
        # up_points, _ = pytorch3d.ops.sample_farthest_points(up_points, K=int(up_ratio*num_points))
        # up_points = up_points.cpu()

        down_idx = farthest_point_sample(up_points, up_ratio*num_points)
        up_points = index_points(up_points, down_idx)
        up_points = up_points.cpu()
    
    if 1:
        # 首先找到与预测点一一对应的原始点
        idx =  idx.reshape(1, -1, 1)   # [1, 29952, 1]
        new_idx = index_points(idx, down_idx.long()).squeeze(2)   # [1, N, 1] --> [1, N]
        points = torch.from_numpy(points).unsqueeze(0)  # [1, N, 3]
        old_points = index_points(points, new_idx.long())  # [1, N, 3]
        up_points = get_correction_points(old_points, up_points)

    up_points = up_points[0, ...]   # [rN, 3]
    up_points = up_points.numpy() * all_radius + all_centroid

    return up_points


def large_pts_denoise(patch_size, net, points, num_patch, device, batch_size=8):
    

    if len(points) >= patch_size*2:
        xyz_list = pts_split(points, patch_size)
        denosie_list = []
        for xyz in xyz_list:
            denoise_xyz = optimize_once(net, xyz, num_patch, device, batch_size)
            denosie_list.append(denoise_xyz)
        denoise_points = np.concatenate(denosie_list, axis=0)
    else:
        return optimize_once(net, points, num_patch, device, batch_size)

    return denoise_points

def denoisingX(iteration, net, ply_path, num_patch, device, batch_size, intermediate=False):
    
    points = read_ply(ply_path)[:, 0:3]
    mode_name = os.path.splitext(ply_path)[0]
    for iter in range(iteration):
        points_num = points.shape[0]
        # points = optimize_once(net, points, num_patch, device, batch_size)
        points = large_pts_denoise(10000, net, points, num_patch, device, batch_size)
        
        save_path = mode_name + "_i%d.ply"%(iter+1)
        if intermediate:
            save_ply(points, save_path)
        print("iter%d result: %d --> %d"%(iter, points_num, points.shape[0]))
    if not intermediate:
        save_ply(points, save_path)


def upsamplingX(up_ratio, iteration, net, ply_path, num_patch, device, batch_size, intermediate=False):

    mode_name = os.path.splitext(ply_path)[0]
    
    points = read_ply(ply_path)[:, 0:3]
    points_num = len(points)
    points = upsample_once(up_ratio, net, points, num_patch, device, batch_size)
    if intermediate:
        save_path = mode_name + "_X%d.ply"%(up_ratio)
        save_ply(points, save_path)
    print("upsampleX%d result: %d --> %d"%(up_ratio, points_num, points.shape[0]))

    refine_iter = iteration
    for iter in range(refine_iter):
        points_num = len(points)
        points = optimize_once(net, points, num_patch, device, batch_size)
        save_path = mode_name + "_X%d_i%d.ply"%(up_ratio, iter+1)
        if intermediate:
            save_ply(points, save_path)
        print("iter%d result: %d --> %d"%(iter, points_num, points.shape[0]))
    if not intermediate:
        save_ply(points, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('TPUNet')
    parser.add_argument('--ply_path', type=str, default="data/upsample_data/genus3n30e-3.ply", help='the file of test data')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size in test')   
    parser.add_argument('--pretrained_weights', type=str, default='pretrained/pretrained_model_new.pth', help='pretrain models file')  # 416
    parser.add_argument('--iteration', type=int, default=10, help='pretrain models file')  # 416
    parser.add_argument('--up_ratio', type=int, default=4, help='pretrain models file')  # 416
    parser.add_argument('--task', type=str, default="upsample", help='denoise/upsample')  # 416

    args = parser.parse_args()

    device = torch.device("cuda:0")
    net = PointDenoising().to(device)
    net.load_state_dict(torch.load(args.pretrained_weights, map_location=device), strict=True)
    print("load model: {}".format(args.pretrained_weights))

    if args.task == "denoise":
        denoisingX(args.iteration, net, args.ply_path, DENOISE.mini_point, device, args.batch_size, intermediate=True)
    elif args.task == "upsample":
        upsamplingX(args.up_ratio, args.iteration, net, args.ply_path, DENOISE.mini_point, device, args.batch_size, intermediate=True)










