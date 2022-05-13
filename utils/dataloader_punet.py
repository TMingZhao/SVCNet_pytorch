import random
import os
import math
from scipy.misc import face
import torch
import numbers
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm.auto import tqdm
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from utils.point_util import index_points
from utils import operations
from utils.pc_utils import read_ply, save_ply, load_off


import importlib
P3D_found = importlib.util.find_spec("pytorch3d") is not None
if P3D_found:
    import pytorch3d.ops

class NormalizeUnitSphere(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def normalize(pcl, center=None, scale=None):
        """
        Args:
            pcl:  The point cloud to be normalized, (N, 3)
        """
        if center is None:
            p_max = pcl.max(dim=0, keepdim=True)[0]
            p_min = pcl.min(dim=0, keepdim=True)[0]
            center = (p_max + p_min) / 2    # (1, 3)
        pcl = pcl - center
        if scale is None:
            scale = (pcl ** 2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]  # (1, 1)
        pcl = pcl / scale
        return pcl, center, scale

    def normalize_point_batch(self, pc, NCHW):
        """
        normalize a batch of point clouds
        :param
            pc      [B, N, 3] or [B, 3, N]
            NCHW    if True, treat the second dimension as channel dimension
        :return
            pc      normalized point clouds, same shape as input
            centroid [B, 1, 3] or [B, 3, 1] center of point clouds
            furthest_distance [B, 1, 1] scale of point clouds
        """
        point_axis = 2 if NCHW else 1
        dim_axis = 1 if NCHW else 2
        centroid = torch.mean(pc, dim=point_axis, keepdim=True)
        pc = pc - centroid
        furthest_distance, _ = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=dim_axis, keepdim=True)), dim=point_axis, keepdim=True)
        # 避免出现除数等于0
        furthest_distance = torch.where(furthest_distance==0, torch.ones_like(furthest_distance), furthest_distance)
        pc = pc / furthest_distance
        return pc, centroid, furthest_distance

    def __call__(self, data, isBatchdata=False):
        
        
        if isBatchdata:
            data['pcl_clean'], center, scale = self.normalize_point_batch(data['pcl_clean'], NCHW=True)
            data['pcl_noisy'] = (data['pcl_noisy'] - center) / scale
            if 'edge' in data:
                data['edge'][:, 0:3] = (data['edge'][:, 0:3] - center) / scale
                data['edge'][:, 3:6] = (data['edge'][:, 3:6] - center) / scale
        else:
            # assert 'pcl_noisy' not in data, 'Point clouds must be normalized before applying noise perturbation.'
            data['pcl_clean'], center, scale = self.normalize(data['pcl_clean'])
            if 'pcl_noisy' in data:
                data['pcl_noisy'], center, scale = self.normalize(data['pcl_noisy'], center, scale )
            if 'vert' in data:
                data['vert'], center, scale = self.normalize(data['vert'], center, scale )
            if 'edge' in data:
                data['edge'][:, 0:3], center, scale = self.normalize(data['edge'][:, 0:3], center, scale )
                data['edge'][:, 3:6], center, scale = self.normalize(data['edge'][:, 3:6], center, scale )
        data['center'] = center
        data['scale'] = scale
        return data


class AddNoise(object):

    def __init__(self, noise_std_min, noise_std_max):
        super().__init__()
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

    def __call__(self, data):
        # bts = data['pcl_clean'].shape[0]
        # noise_std = self.noise_std_min + (self.noise_std_max - self.noise_std_min ) * torch.rand((bts, 1, 1))
        noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        if 'pcl_noisy' not in data:
            data['pcl_noisy'] = data['pcl_clean'] + torch.randn_like(data['pcl_clean']) * noise_std
        else:
            data['pcl_noisy'] = data['pcl_noisy'] + torch.randn_like(data['pcl_noisy']) * noise_std
        data['noise_std'] = noise_std
        return data


class AddLaplacianNoise(object):

    def __init__(self, noise_std_min, noise_std_max):
        super().__init__()
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

    def __call__(self, data):
        noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        noise = torch.FloatTensor(np.random.laplace(0, noise_std, size=data['pcl_clean'].shape)).to(data['pcl_clean'])
        data['pcl_noisy'] = data['pcl_clean'] + noise
        data['noise_std'] = noise_std
        return data


class AddUniformBallNoise(object):
    
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def __call__(self, data):
        N = data['pcl_clean'].shape[0]
        phi = np.random.uniform(0, 2*np.pi, size=N)
        costheta = np.random.uniform(-1, 1, size=N)
        u = np.random.uniform(0, 1, size=N)
        theta = np.arccos(costheta)
        r = self.scale * u ** (1/3)

        noise = np.zeros([N, 3])
        noise[:, 0] = r * np.sin(theta) * np.cos(phi)
        noise[:, 1] = r * np.sin(theta) * np.sin(phi)
        noise[:, 2] = r * np.cos(theta)
        noise = torch.FloatTensor(noise).to(data['pcl_clean'])
        data['pcl_noisy'] = data['pcl_clean'] + noise
        return data


class AddCovNoise(object):

    def __init__(self, cov, std_factor=1.0):
        super().__init__()
        self.cov = torch.FloatTensor(cov)
        self.std_factor = std_factor

    def __call__(self, data):
        num_points = data['pcl_clean'].shape[0]
        noise = np.random.multivariate_normal(np.zeros(3), self.cov.numpy(), num_points) # (N, 3)
        noise = torch.FloatTensor(noise).to(data['pcl_clean'])
        data['pcl_noisy'] = data['pcl_clean'] + noise * self.std_factor
        data['noise_std'] = self.std_factor
        return data


class AddDiscreteNoise(object):

    def __init__(self, scale, prob=0.1):
        super().__init__()
        self.scale = scale
        self.prob = prob
        self.template = np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ], dtype=np.float32)

    def __call__(self, data):
        num_points = data['pcl_clean'].shape[0]
        uni_rand = np.random.uniform(size=num_points)
        noise = np.zeros([num_points, 3])
        for i in range(self.template.shape[0]):
            idx = np.logical_and(0.1*i <= uni_rand, uni_rand < 0.1*(i+1))
            noise[idx] = self.template[i].reshape(1, 3)
        noise = torch.FloatTensor(noise).to(data['pcl_clean'])
        # print(data['pcl_clean'])
        # print(self.scale)
        data['pcl_noisy'] = data['pcl_clean'] + noise * self.scale
        data['noise_std'] = self.scale
        return data


class RandomScale(object):

    def __init__(self, scales):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def __call__(self, data):
        # min_scale , max_scale = self.scales
        # bts = data['pcl_clean'].shape[0]
        # scale = min_scale + (max_scale- min_scale ) * torch.rand((bts, 1, 1))
        scale = random.uniform(*self.scales)
        data['pcl_clean'] = data['pcl_clean'] * scale
        if 'pcl_noisy' in data:
            data['pcl_noisy'] = data['pcl_noisy'] * scale
        if 'vert' in data:
            data['vert'] = data['vert'] * scale
        if 'edge' in data:
            data['edge'] = data['edge'] * scale
        return data


class RandomRotate(object):

    def __init__(self, degrees=180.0, axis=0):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        matrix = torch.tensor(matrix)

        data['pcl_clean'] = torch.matmul(data['pcl_clean'], matrix)
        if 'pcl_noisy' in data:
            data['pcl_noisy'] = torch.matmul(data['pcl_noisy'], matrix)
        if 'vert' in data:
            data['vert'] = torch.matmul(data['vert'], matrix)
        if 'edge' in data:
            data['edge'][:, 0:3] = torch.matmul(data['edge'][:, 0:3], matrix)
            data['edge'][:, 3:6] = torch.matmul(data['edge'][:, 3:6], matrix)

        return data

def standard_train_transforms(noise_std_min, noise_std_max, scale_d=0.2, rotate=True):
    transforms = [
        NormalizeUnitSphere(),
        AddNoise(noise_std_min=noise_std_min, noise_std_max=noise_std_max),
        RandomScale([1.0-scale_d, 1.0+scale_d]),
    ]
    if rotate:
        transforms += [
            RandomRotate(axis=0),
            RandomRotate(axis=1),
            RandomRotate(axis=2),
        ]
    return Compose(transforms)

class PointCloudDataset(Dataset):

    def __init__(self, root, dataset, split, resolution, transform=None):
        super().__init__()
        self.pcl_dir = os.path.join(root, dataset, 'pointclouds', split, resolution)
        self.mesh_dir = os.path.join(root, dataset, 'meshes', split)
        self.transform = transform
        self.pointclouds = []
        self.pointcloud_names = []
        self.verts = []
        self.faces = []
        for fn in tqdm(os.listdir(self.pcl_dir), desc='Loading'):
            if fn[-3:] != 'ply':
                continue
            pcl_path = os.path.join(self.pcl_dir, fn)
            if not os.path.exists(pcl_path):
                raise FileNotFoundError('File not found: %s' % pcl_path)
            pcl = torch.FloatTensor(read_ply(pcl_path))
            self.pointclouds.append(pcl)
            self.pointcloud_names.append(fn[:-4])
            mesh_path = os.path.join(self.mesh_dir, fn[:-4]+".off")
            if not os.path.exists(mesh_path):
                raise FileNotFoundError('File not found: %s' % mesh_path)
            vert, face = load_off(mesh_path)
            vert = torch.FloatTensor(vert)
            face = torch.LongTensor(face)
            self.verts.append(vert)
            self.faces.append(face)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {
            'pcl_clean': self.pointclouds[idx].clone(), 
            'name': self.pointcloud_names[idx],
            'vert': self.verts[idx].clone(),
            'face': self.faces[idx].clone(),
        }
        if self.transform is not None:
            data = self.transform(data)
        return data


def make_patches_for_pcl_pair(pcl_A, pcl_B, patch_size, num_patches, ratio=1):
    """
    Args:
        pcl_A:  The first point cloud, (N, 3).
        pcl_B:  The second point cloud, (rN, 3).
        patch_size:   Patch size M.
        num_patches:  Number of patches P.
        ratio:    Ratio r.
    Returns:
        (P, M, 3), (P, rM, 3)
    """
    if P3D_found:
        N = pcl_A.size(0)
        seed_idx = torch.randperm(N)[:num_patches]   # (P, )
        seed_pnts = pcl_A[seed_idx].unsqueeze(0)   # (1, P, 3)
        _, _idx, pat_A = pytorch3d.ops.knn_points(seed_pnts, pcl_A.unsqueeze(0), K=patch_size, return_nn=True)
        pat_A = pat_A[0]    # (P, M, 3)
        # _, _, pat_B = pytorch3d.ops.knn_points(seed_pnts, pcl_B.unsqueeze(0), K=int(ratio*patch_size), return_nn=True)
        pat_B = pytorch3d.ops.masked_gather(pcl_B.unsqueeze(0), _idx)
        pat_B = pat_B[0]
    else:
        pcl_A = pcl_A.unsqueeze(0)  # [1, N, 3]
        pcl_B = pcl_B.unsqueeze(0)  # [1, N, 3]
        # 生成随机种子点
        rand_idx = np.random.randint(0, pcl_A.shape[1], size=(1, num_patches))   # [1, P]
        rand_idx = torch.from_numpy(rand_idx)   # [1, P]
        seed_pt = index_points(pcl_A, rand_idx)    # [1, P, 3]

        pat_A, index, _ = operations.group_knn(patch_size, seed_pt, pcl_A, NCHW=False)    # [1, P, K, 3] [1, B, K]
        pat_A = pat_A.squeeze(0) # [P, K, 3]
        # 选取相同位置的噪声点
        # operations.gather_points()
        pat_B = index_points(pcl_B, index)    # [1, P, K, 3]
        pat_B = pat_B.squeeze(0) # [P, K, 3]


    return pat_A, pat_B

class PairedPatchDataset(Dataset):
    def __init__(self, datasets, patch_ratio, on_the_fly=True, patch_size=32, num_patches=1000, mini_batch=1, transform=None):
        super().__init__()
        self.datasets = datasets
        self.len_datasets = sum([len(dset) for dset in datasets])
        self.patch_ratio = patch_ratio
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.sub_batch_size = mini_batch
        self.on_the_fly = on_the_fly
        self.transform = transform
        self.patches = []
        # Initialize
        if not on_the_fly:
            self.make_patches()

    def make_patches(self):
        for dataset in tqdm(self.datasets, desc='MakePatch'):
            for data in tqdm(dataset):
                pat_noisy, pat_clean = make_patches_for_pcl_pair(
                    data['pcl_noisy'],
                    data['pcl_clean'],
                    patch_size=self.patch_size,
                    num_patches=self.num_patches,
                    ratio=self.patch_ratio
                )   # (P, M, 3), (P, rM, 3)
                for i in range(pat_noisy.size(0)):
                    self.patches.append((pat_noisy[i], pat_clean[i], ))

    def __len__(self):
        if not self.on_the_fly:
            return len(self.patches)
        else:
            return self.len_datasets * self.num_patches


    def __getitem__(self, idx):
        if self.on_the_fly:
            pcl_dset = random.choice(self.datasets)
            pcl_data = pcl_dset[idx % len(pcl_dset)]
            pat_noisy, pat_clean = make_patches_for_pcl_pair(
                pcl_data['pcl_noisy'],
                pcl_data['pcl_clean'],
                patch_size=self.patch_size,
                num_patches=self.sub_batch_size,
                ratio=self.patch_ratio
            )
            data = {
                'pcl_noisy': pat_noisy.permute(0, 2, 1),
                'pcl_clean': pat_clean.permute(0, 2, 1),
                'vert': pcl_data['vert'],
                'face': pcl_data['face']
            }
        else:
            data = {
                'pcl_noisy': self.patches[idx][0].clone(), 
                'pcl_clean': self.patches[idx][1].clone(),
            }
        if self.transform is not None:
            data = self.transform(data, True)
        return data

    def collate_fn(self, batch):

        batch_pcl_noisy = []
        batch_pcl_clean =[]
        batch_pos = []
        batch_mesh = []
        for batch_i in batch:
            pcl_noisy = batch_i['pcl_noisy']
            pcl_clean = batch_i['pcl_clean']
            vert = batch_i['vert']
            face = batch_i['face']
            pcl_center = batch_i['center']
            pcl_scale = batch_i['scale']
            pos = torch.cat([pcl_center, pcl_scale], dim=1)
            batch_mesh.append((vert, face))
            batch_pcl_noisy.append(pcl_noisy)
            batch_pcl_clean.append(pcl_clean)
            batch_pos.append(pos)

        batch_pcl_noisy = torch.cat(batch_pcl_noisy, 0)
        batch_pcl_clean = torch.cat(batch_pcl_clean, 0)
        batch_pos = torch.cat(batch_pos, 0)


# input_points, patch_pos, clear_points, gt_points, mesh, whole_pos 
        return batch_pcl_noisy, batch_pos, batch_pcl_clean, batch_pcl_clean, batch_mesh, batch_pos


def ply_visual(plys_in, plys_gt, path=None, unis=None):
    # plys_in : [B, 3, N]
    # plys_gt : [B, 3, 2N]
    plys_in = plys_in.permute(0, 2, 1)
    plys_gt = plys_gt.permute(0, 2, 1)
    for idx, (ply_in, ply_gt) in enumerate(zip(plys_in, plys_gt)):
        save_merge_xyz_color(ply_in, ply_gt, "output/visual/batch_%d.ply"%(idx))

if __name__ == "__main__":
    from utils.pc_utils import save_merge_xyz_color, save_merge_batch_xyz_color
    train_dset = PairedPatchDataset(
        datasets=[
            PointCloudDataset(
                root="data",
                dataset="PUNet",
                split='train',
                resolution=resl,
                transform=standard_train_transforms(noise_std_max=0.020, noise_std_min=0.005, rotate=True)
            ) for resl in ['5000_poisson']  # , '10000_poisson', '30000_poisson', '50000_poisson'
        ],
        patch_size=32,
        patch_ratio=1,
        mini_batch=10,
        on_the_fly=True,
        transform= NormalizeUnitSphere()
    )
    dataloader = torch.utils.data.DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_dset.collate_fn,)
    for batch in tqdm(dataloader):
        input_points, patch_pos, clear_points, gt_points, mesh, whole_pos = batch
        save_merge_batch_xyz_color(input_points, clear_points, "output/visual/batch_in_gt.ply")
        for i,(input_i, mesh_i, pos) in enumerate(zip(clear_points, mesh, patch_pos)):
            pcl_center = pos[0:3, :]
            pcl_scale = pos[3:4, :]
            input_i = input_i * pcl_scale + pcl_center
            vert, face_ = mesh_i
            save_merge_xyz_color(input_i.permute(1, 0), vert, "output/visual/batch_%d_vert.ply"%(i))
        # pcl_noisy = batch['pcl_noisy']
        # pcl_clean = batch['pcl_clean']
        ply_visual(input_points, clear_points)

