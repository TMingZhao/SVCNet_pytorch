import h5py
from torch.utils.data import Dataset
import numpy as np
import torch
import sys
import os
from tqdm import tqdm
sys.path.append(os.getcwd())
from utils.pc_utils import save_merge_batch_xyz_color, save_merge_xyz_color, downsample_points, save_ply, save_merge_xyz
from utils.dataloader_punet import standard_train_transforms, make_patches_for_pcl_pair, NormalizeUnitSphere
NUM_EDGE = 120
NUM_FACE = 800
from utils.point_util import farthest_point_sample, index_points

def load_patch_data(skip_rate = 1):
    h5_filename = 'data/ECNet_data/mix_CAD1k_halfnoise_1.h5'
    f = h5py.File(h5_filename, "r")
    input = f['mc8k_input'][:]
    dist = f['mc8k_dist'][:]
    edge = f['edge'][:]
    edgepoint = f['edge_points'][:]
    face = f['face'][:]
    gt = f['gt'][:]

    assert edge.shape[1]==NUM_EDGE
    assert face.shape[1]==NUM_FACE
    edge = np.reshape(edge,(-1,NUM_EDGE*2,3))
    face = np.reshape(face,(-1,NUM_FACE*3,3))
    edge = np.concatenate([edge,face,edgepoint],axis=1)
    name = f['name'][:]
    assert len(input) == len(edge)

    # save_ply(face[0], "output/visual/0_face.ply")
    # save_merge_xyz_color(input[0], gt[0], "output/visual/0_in_face.ply")
    # ####
    h5_filename = 'data/ECNet_data/mix_Virtualscan1k_halfnoise_1.h5'
    f = h5py.File(h5_filename, "r")
    input1 = f['mc8k_input'][:]
    dist1 = f['mc8k_dist'][:]
    edge1 = f['edge'][:]
    edgepoint1 = f['edge_points'][:]
    face1 = f['face'][:]
    gt1 = f['gt'][:]
    assert edge1.shape[1] == NUM_EDGE
    assert face1.shape[1] == NUM_FACE
    edge1 = np.reshape(edge1, (-1, NUM_EDGE * 2, 3))
    face1 = np.reshape(face1, (-1, NUM_FACE * 3, 3))
    edge1 = np.concatenate([edge1, face1, edgepoint1], axis=1)
    name1 = f['name'][:]
    assert len(input1) == len(edge1)
    input = np.concatenate([input,input1],axis=0)
    gt = np.concatenate([gt,gt1],axis=0)
    dist  = np.concatenate([dist,dist1],axis=0)
    edge  = np.concatenate([edge,edge1],axis=0)
    name = np.concatenate([name,name1])
    # ######

    data_radius = np.ones(shape=(len(input)))
    centroid = np.mean(input[:,:,0:3], axis=1, keepdims=True)
    input[:,:,0:3] = input[:,:,0:3] - centroid
    distance = np.sqrt(np.sum(input[:,:,0:3] ** 2, axis=-1))
    furthest_distance = np.amax(distance,axis=1,keepdims=True)
    input[:, :, 0:3] = input[:,:,0:3] / np.expand_dims(furthest_distance,axis=-1)
    

    dist = dist/furthest_distance

    gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
    gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance,axis=-1)

    edge[:, :, 0:3] = edge[:, :, 0:3] - centroid
    edge[:, :, 0:3] = edge[:, :, 0:3] / np.expand_dims(furthest_distance,axis=-1)

    input = input[::skip_rate]
    dist = dist[::skip_rate]
    edge = edge[::skip_rate]
    name = name[::skip_rate]
    data_radius = data_radius[::skip_rate]
    gt = gt[::skip_rate]
    
    # object_name = []
    # for item in name:
    #     str_1 = item.split('/')[-1]
    #     str_2 = str_1.split('_')[0]
    #     object_name.append(str_2)
    # object_name = list(set([item.split('/')[-1].split('_')[0] for item in name]))
    # object_name.sort()
    # print( "load object names {}".format(object_name))
    # print( "total %d samples" % (len(input)))
    return input, gt, dist, edge, data_radius, name



class ECDataDataset(Dataset):

    def __init__(self, patch_size=32, sub_batch_size=4, transform=None):
        super().__init__()
        self.transform = transform
        input_, gt, dist, edge, data_radius, self.name = load_patch_data()

        self.patch_size = patch_size
        self.sub_batch_size = sub_batch_size
        self.patch_norm = NormalizeUnitSphere()

        input_ = torch.from_numpy(input_)
        gt = torch.from_numpy(gt)
        # down_idx = farthest_point_sample(gt, 50) # [1200, 32]
        # self.input_points = index_points(input_, down_idx)  # [1200, 32, 3]
        # self.gt_points = index_points(gt, down_idx)  # [1200, 32, 3]

        down_idx = farthest_point_sample(input_, 80) # [1200, 32]
        self.models = index_points(input_, down_idx)  # [1200, 32, 3]
        # self.input_points = input_  # [1200, 1024, 3]
        # self.gt_points = gt  # [1200, 1024, 3]
        edge = np.reshape(edge[:,0:2*NUM_EDGE,:],(-1, NUM_EDGE,6))
        self.edge_points = torch.FloatTensor(edge)  # [1200, 120, 6]


    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        # idx = 35
        data = {
            # 'pcl_noisy': self.input_points[idx].clone(), 
            'pcl_clean': self.models[idx].clone(), 
            'name': self.name[idx],
            'edge': self.edge_points[idx].clone(),
        }
        if self.transform is not None:
            data = self.transform(data)

        pat_noisy, pat_clean = make_patches_for_pcl_pair(
                data['pcl_noisy'],
                data['pcl_clean'],
                patch_size=self.patch_size,
                num_patches=self.sub_batch_size,
                ratio=1
            )
        patch_edge = data['edge'].unsqueeze(0).repeat(self.sub_batch_size, 1, 1)  # [B, N, C]
        data = {
            'pcl_noisy': pat_noisy.permute(0, 2, 1),
            'pcl_clean': pat_clean.permute(0, 2, 1),
            'edge': patch_edge.permute(0, 2, 1),
        }
        data = self.patch_norm(data, True)

        return data

    def collate_fn(self, batch):

        batch_pcl_noisy = []
        batch_pcl_clean =[]
        batch_pos = []
        batch_mesh = []
        batch_edge = []
        for batch_i in batch:
            pcl_noisy = batch_i['pcl_noisy']
            pcl_clean = batch_i['pcl_clean']
            edge = batch_i['edge']
            pcl_center = batch_i['center']
            pcl_scale = batch_i['scale']
            pos = torch.cat([pcl_center, pcl_scale], dim=1)
            batch_pcl_noisy.append(pcl_noisy)
            batch_pcl_clean.append(pcl_clean)
            batch_pos.append(pos)
            batch_edge.append(edge)

        batch_pcl_noisy = torch.cat(batch_pcl_noisy, 0)
        batch_pcl_clean = torch.cat(batch_pcl_clean, 0)
        batch_pos = torch.cat(batch_pos, 0)
        batch_edge = torch.cat(batch_edge, 0)

        return batch_pcl_noisy, batch_pos, batch_pcl_clean, batch_pcl_clean, batch_mesh, batch_edge


def distance_point2edge_np(points, edges):
    # points: [B, N, 3] 
    # edges: [B, M, 6]
    segment0 = edges[:, :, 0:3]
    segment1 = edges[:, :, 3:6]
    points = np.expand_dims(points, axis=2)
    segment0 = np.expand_dims(segment0, axis=1)
    segment1 = np.expand_dims(segment1, axis=1)

    v = segment1 - segment0     # (1, 1, 120, 3)
    w = points - segment0       # (1, 1024, 120, 3)

    c1 = np.sum(w * v, axis=-1) # (1, 1024, 120)  这里是点乘
    c2 = np.sum(v * v, axis=-1) # (1, 1, 120)

    # distance to the line
    distance0 = np.sum(np.power(points - segment0, 2), axis=-1) # (1024, 1024, 120)
    distance1 = np.sum(np.power(points - segment1, 2), axis=-1) # (1, 1024, 120)
    b = c1 / c2     # (1, 1024, 120)
    b = np.expand_dims(b, axis=-1)  # (1, 1024, 120, 1)
    segmentb = segment0 + b * v       # (1, 1024, 120, 3)
    distanceb = np.sum(np.power(points - segmentb, 2), axis=-1)     # (1, 1024, 120)
    dist = np.where(c2 <= c1, distance1, distanceb)     # (1, 1024, 120)
    dist = np.where(c1 <= 0, distance0, dist)        # (1, 1024, 120)
    return dist

def distance_point2edge_tensor(points, edges):
    # points: [B, N, 3] 
    # edges: [B, M, 6]
    segment0 = edges[:, :, 0:3]
    segment1 = edges[:, :, 3:6]
    points = points.unsqueeze(2)
    segment0 = segment0.unsqueeze(1)
    segment1 = segment1.unsqueeze(1)

    v = segment1 - segment0
    w = points - segment0

    c1 = torch.sum(w * v, dim=-1, keepdim=False)    # (1, 1024, 120)
    c2 = torch.sum(v * v, dim=-1, keepdim=False)    # (1, 1, 120)

    # distance to the line
    distance0 = torch.sum((points - segment0)**2, dim=-1, keepdim=False)    # [1, 1024, 120])
    distance1 = torch.sum((points - segment1)**2, dim=-1, keepdim=False)    # [1, 1024, 120])
    b = c1 / c2    # (1, 1024, 120)
    b = b.unsqueeze(-1)    # (1, 1024, 120, 1)
    segmentb = segment0 + b * v    # (1, 1024, 120, 3)
    distanceb = torch.sum((points - segmentb)**2, dim=-1, keepdim=False)    # (1, 1024, 120)
    dist = torch.where(c2 <= c1, distance1, distanceb)
    dist = torch.where(c1 <= 0, distance0, dist)    # [B, N, M])    # 得到每个点到边缘点的距离

    point2edge, idx = torch.min(dist, dim=-1, keepdim=False) # [B, N]   # [B, N]
    return point2edge, idx


if __name__ == "__main__":

    input, gt, dist, edge, data_radius, name = load_patch_data()
    # for i, (input_i, edge_i) in enumerate(zip(input, edge)):
        
    #     print(i)
    #     edge_points = np.reshape(edge_i[0:2*NUM_EDGE,:],(NUM_EDGE,6))
    #     dist_1 = distance_point2edge_np(np.expand_dims(input_i, 0), np.expand_dims(edge_points, 0))

    #     dist_1_t = distance_point2edge_tensor(torch.from_numpy(np.expand_dims(input_i, 0)), torch.from_numpy(np.expand_dims(edge_points, 0)))
    #     dist_1_t_n = dist_1_t.numpy()
    #     long_edge_point = edge_i[2640::, :]
    #     dist_2 = distance_point2edge_np(np.expand_dims(long_edge_point, 0), np.expand_dims(edge_points, 0))
    #     idx, dist_2 = torch.min(torch.from_numpy(dist_2), dim=-1, keepdim=False)

    #     segment0 = edge_points[:, 0:3]
    #     segment1 = edge_points[:, 3:6]
    #     save_merge_xyz_color(torch.from_numpy(input_i), torch.from_numpy(segment0), "output/visual/1.ply")
    #     save_merge_xyz_color(torch.from_numpy(input_i), torch.from_numpy(segment1), "output/visual/2.ply")
    #     save_merge_xyz_color(torch.from_numpy(input_i), torch.cat([torch.from_numpy(segment0), torch.from_numpy(segment1)], dim=0), "output/visual/3.ply")

    dataset = ECDataDataset(patch_size=32, transform=standard_train_transforms(noise_std_max=0.10, noise_std_min=0.02, rotate=True))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=1, pin_memory=True, collate_fn=dataset.collate_fn,)
    for batch in tqdm(dataloader):
        input_points, patch_pos, clear_points, gt_points, mesh, edge = batch
        save_merge_batch_xyz_color(input_points, edge[:,0:3,:], "output/visual/batch_in_edge.ply")
        save_merge_batch_xyz_color(input_points, clear_points, "output/visual/batch_in_gt.ply")
        save_merge_batch_xyz_color(clear_points, edge[:,0:3,:], "output/visual/batch_gt_edge.ply")
        # for i, (input_i, clean_i, edge_i) in enumerate(zip(input_points, clear_points, edge)):
        #     save_merge_xyz_color(input_i.permute(1,0), clean_i.permute(1,0), "output/visual/in_gt.ply")
        #     segment0 = edge_i[0:3, :]
        #     segment1 = edge_i[3:6, :]
        #     save_merge_xyz_color(clean_i.permute(1,0), segment0, "output/visual/1.ply")

