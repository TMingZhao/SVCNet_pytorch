""" Utility functions for processing point clouds.
"""
import os
import numpy as np
import sys
sys.path.append(os.getcwd())
# Point cloud IO
from matplotlib import cm
import plyfile


import torch
import point_cloud_utils as pcu



def normalize_point_cloud(input):
    """
    input: pc [N, P, 3]
    output: pc, centroid, furthest_distance   [B, N, 3]  [B, 1, 3]   [B, 1, 1]
    """
    if len(input.shape) == 2:
        axis = 0
    elif len(input.shape) == 3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(input ** 2, axis=-1, keepdims=True)), axis=axis, keepdims=True)
    # # 避免出现除数等于0
    # furthest_distance = np.where(furthest_distance==0, np.ones_like(furthest_distance), furthest_distance)
    input = input / furthest_distance
    return input, centroid, furthest_distance


def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02, is_2D=False):
    """
    Randomly jitter points. jittering is per point.
    Input:
        batch_data: BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    chn = 2 if is_2D else 3
    # jittered_data = np.clip(sigma * np.random.randn(B, N, C, dtype=batch_data.dtype), -clip, clip)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C).astype(batch_data.dtype), -clip, clip)
    jittered_data[:, :, chn:] = 0
    jittered_data += batch_data
    return jittered_data

def jitter_perturbation_point_cloud_with_gaussian(batch_data, sigma=0.005, clip=1, is_2D=False):
    """
    Randomly jitter points. jittering is per point.
    Input:
        batch_data: BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, jittered batch of point clouds
    """

    norm_data, centroid, furthest_distance =  normalize_point_cloud(batch_data) # [B, N, 3]  [B, 1, 3]   [B, 1, 1]

    B, N, C = batch_data.shape
    assert(clip > 0)
    chn = 2 if is_2D else 3
    # jittered_data = np.clip(np.random.normal(0, sigma, size=(B, N, C)).astype(norm_data.dtype), -clip, clip)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C).astype(norm_data.dtype), -clip, clip)

    jittered_data[:, :, chn:] = 0
    jittered_data[:, :, 2] = 0
    jittered_data += norm_data

    jittered_data = jittered_data * furthest_distance + centroid

    return jittered_data

def jitter_perturbation_point_cloud_with_multiplicative_noise(batch_data, sigma=0.05, clip=1, is_2D=False):
    """
    Randomly jitter points. jittering is per point.
    Input:
        batch_data: BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, jittered batch of point clouds
    """

    norm_data, centroid, furthest_distance =  normalize_point_cloud(batch_data) # [B, N, 3]  [B, 1, 3]   [B, 1, 1]

    B, N, C = batch_data.shape
    assert(clip > 0)
    chn = 2 if is_2D else 3
    min_ = 1 - sigma
    max_ = 1 + sigma
    jittered_data = np.random.rand(*(B, N, C)).astype(norm_data.dtype) * 2 * sigma + min_   # 生成0.95-1.05之间的随机数
    jittered_data[:, :, chn:] = 1
    jittered_data = norm_data * jittered_data

    jittered_data = jittered_data * furthest_distance + centroid

    return jittered_data

# 注意：这个代码会改变输入batch_data等的值!!!
def rotate_point_cloud_and_gt(batch_data, batch_gt=None, batch_gt_1=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    for k in range(batch_data.shape[0]):
        angles = np.random.uniform(size=(3)) * 2 * np.pi
        # angles = np.array([0.76541452, 0.08133027, 0.79327904]) * 2 * np.pi
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]], dtype=batch_data.dtype)
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]], dtype=batch_data.dtype)
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]], dtype=batch_data.dtype)
        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

        batch_data[k, ..., 0:3] = np.dot(
            batch_data[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
        if batch_data.shape[-1] > 3:
            batch_data[k, ..., 3:] = np.dot(
                batch_data[k, ..., 3:].reshape((-1, 3)), rotation_matrix)


        if batch_gt is not None:
            batch_gt[k, ..., 0:3] = np.dot(
                batch_gt[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
            if batch_gt.shape[-1] > 3:
                batch_gt[k, ..., 3:] = np.dot(
                    batch_gt[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

        if batch_gt_1 is not None:
            batch_gt_1[k, ..., 0:3] = np.dot(
                batch_gt_1[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
            if batch_gt_1.shape[-1] > 3:
                batch_gt_1[k, ..., 3:] = np.dot(
                    batch_gt_1[k, ..., 3:].reshape((-1, 3)), rotation_matrix)


    return batch_data, batch_gt, batch_gt_1


def random_scale_point_cloud_and_gt(batch_data, batch_gt=None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, (B, 1, 1)).astype(batch_data.dtype)

    batch_data = np.concatenate([batch_data[:, :, :3] * scales, batch_data[:, :, 3:]], axis=-1)

    if batch_gt is not None:
        batch_gt = np.concatenate([batch_gt[:, :, :3] * scales, batch_gt[:, :, 3:]], axis=-1)

    return batch_data, batch_gt, np.squeeze(scales)


def downsample_points(pts, K):
    # if num_pts > 8K use farthest sampling
    # else use random sampling
    if pts.shape[0] >= 2 * K:
        sampler = FarthestSampler()
        return sampler(pts, K)
    else:
        return pts[np.random.choice(pts.shape[0], K,
                                    replace=(K < pts.shape[0])), :]


class FarthestSampler:
    def __init__(self):
        pass

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts, k):
        farthest_pts = np.zeros((k, 3), dtype=np.float32)
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        # farthest_pts[0] = pts[0]
        distances = self._calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(
                distances, self._calc_distances(farthest_pts[i], pts))
        return farthest_pts


def read_ply_with_color(file, count=None):
    loaded = plyfile.PlyData.read(file)
    points = np.vstack([loaded['vertex'].data['x'],
                        loaded['vertex'].data['y'], loaded['vertex'].data['z']])
    if 'nx' in loaded['vertex'].data.dtype.names:
        normals = np.vstack([loaded['vertex'].data['nx'],
                             loaded['vertex'].data['ny'], loaded['vertex'].data['nz']])
        points = np.concatenate([points, normals], axis=0)
    colors = None
    if 'red' in loaded['vertex'].data.dtype.names:
        colors = np.vstack([loaded['vertex'].data['red'],
                            loaded['vertex'].data['green'], loaded['vertex'].data['blue']])
        if 'alpha' in loaded['vertex'].data.dtype.names:
            colors = np.concatenate([colors, np.expand_dims(
                loaded['vertex'].data['alpha'], axis=0)], axis=0)
        colors = colors.transpose(1, 0)
        colors = colors.astype(np.float32) / 255.0

    points = points.transpose(1, 0)
    if count is not None:
        if count > points.shape[0]:
            # fill the point clouds with the random point
            tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
            tmp[:points.shape[0], ...] = points
            tmp[points.shape[0]:, ...] = points[np.random.choice(
                points.shape[0], count - points.shape[0]), :]
            points = tmp
        elif count < points.shape[0]:
            # different to pointnet2, take random x point instead of the first
            # idx = np.random.permutation(count)
            # points = points[idx, :]
            points = downsample_points(points, count)
    return points, colors


def read_ply(file, count=None, return_faces=False):
    if file[-4:] == ".xyz":
        points = np.loadtxt(file).astype(np.float32)
    elif file[-4:] == ".bin":
        points = np.fromfile(file, dtype=np.float32, count=-1).reshape([-1, 4])[:, 0:3]
    else:

        loaded = plyfile.PlyData.read(file)
        points = np.vstack([loaded['vertex'].data['x'],
                            loaded['vertex'].data['y'], 
                            loaded['vertex'].data['z']])
        if return_faces:
            try:
                # if 'vertex_indices' in loaded['face'].data.dtype.names:
                faces = np.vstack(loaded['face'].data['vertex_indices'])
                # if 'red' in loaded['face'].data.dtype.names:
                #     texture = np.vstack([
                #         loaded['face'].data['red'],
                #         loaded['face'].data['green'],
                #         loaded['face'].data['blue'],
                #         loaded['face'].data['alpha'],])
                #     texture = texture.transpose(1, 0)
            except:
                faces = None
        
        
        if 'nx' in loaded['vertex'].data.dtype.names:
            normals = np.vstack([loaded['vertex'].data['nx'],
                                loaded['vertex'].data['ny'], loaded['vertex'].data['nz']])
            points = np.concatenate([points, normals], axis=0)

        points = points.transpose(1, 0)
    
    if count is not None:
        if count > points.shape[0]:
            # fill the point clouds with the random point
            tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
            tmp[:points.shape[0], ...] = points
            tmp[points.shape[0]:, ...] = points[np.random.choice(
                points.shape[0], count - points.shape[0]), :]
            points = tmp
        elif count < points.shape[0]:
            # different to pointnet2, take random x point instead of the first
            # idx = np.random.permutation(count)
            # points = points[idx, :]
            points = downsample_points(points, count)
    
    if return_faces:
        return points, faces
    else:
        return points


def save_ply_with_face_property(points, faces, property, property_max, filename, cmap_name="Set1"):
    face_num = faces.shape[0]
    colors = np.full(faces.shape, 0.5)
    cmap = cm.get_cmap(cmap_name)
    for point_idx in range(face_num):
        colors[point_idx] = cmap(property[point_idx] / property_max)[:3]
    save_ply_with_face(points, faces, filename, colors)


def save_ply_with_face(points, faces, filename, colors=None):
    vertex = np.array([tuple(p) for p in points], dtype=[
                      ('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    faces = np.array([(tuple(p),) for p in faces], dtype=[
                     ('vertex_indices', 'i4', (3, ))])
    descr = faces.dtype.descr
    if colors is not None:
        assert len(colors) == len(faces)
        face_colors = np.array([tuple(c * 255) for c in colors],
                               dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        descr = faces.dtype.descr + face_colors.dtype.descr

    faces_all = np.empty(len(faces), dtype=descr)
    for prop in faces.dtype.names:
        faces_all[prop] = faces[prop]
    if colors is not None:
        for prop in face_colors.dtype.names:
            faces_all[prop] = face_colors[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(
        vertex, 'vertex'), plyfile.PlyElement.describe(faces_all, 'face')], text=False)
    ply.write(filename)


def load(filename, count=None):
    if filename[-4:] == ".ply":
        points = read_ply(filename, count)[:, :3].astype(np.float32)
    else:
        points = np.loadtxt(filename).astype(np.float32)
        if count is not None:
            if count > points.shape[0]:
                # fill the point clouds with the random point
                tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
                tmp[:points.shape[0], ...] = points
                tmp[points.shape[0]:, ...] = points[np.random.choice(
                    points.shape[0], count - points.shape[0]), :]
                points = tmp
            elif count < points.shape[0]:
                # different to pointnet2, take random x point instead of the first
                # idx = np.random.permutation(count)
                # points = points[idx, :]
                points = downsample_points(points, count)
    return points


def save_ply(points, filename, colors=None, normals=None):
    '''
    points: array[N,3]
    '''
    vertex = np.core.records.fromarrays(points.transpose(
        1, 0), names='x, y, z', formats='f4, f4, f4')
    num_vertex = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        vertex_normal = np.core.records.fromarrays(
            normals.transpose(1, 0), names='nx, ny, nz', formats='f4, f4, f4')
        assert len(vertex_normal) == num_vertex
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        assert len(colors) == num_vertex
        if colors.max() <= 1:
            colors = colors * 255
        if colors.shape[1] == 4:
            vertex_color = np.core.records.fromarrays(colors.transpose(
                1, 0), names='red, green, blue, alpha', formats='u1, u1, u1, u1')
        else:
            vertex_color = np.core.records.fromarrays(colors.transpose(
                1, 0), names='red, green, blue', formats='u1, u1, u1')
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(num_vertex, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData(
        [plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def save_ply_property(points, property, filename, property_max=None, normals=None, cmap_name='Set1'):
    point_num = points.shape[0]
    colors = np.full([point_num, 3], 0.5)
    cmap = cm.get_cmap(cmap_name)
    if property_max is None:
        property_max = np.amax(property, axis=0)
    for point_idx in range(point_num):
        colors[point_idx] = cmap(property[point_idx] / property_max)[:3]
    save_ply(points, filename, colors, normals)

# 该读取方式极其耗时
def load_off_own(path):

    data = np.loadtxt(path, skiprows=1, max_rows=1, usecols=(0, 1))
    vert_num = int(data[0])
    face_num = int(data[1])
    verts = np.loadtxt(path, skiprows=2, max_rows=vert_num, usecols=(0, 1, 2))
    faces = np.loadtxt(path, skiprows=2+vert_num, max_rows=face_num, usecols=(1, 2, 3))

    return verts, faces

def load_off(path):
    assert(path[-3:] == 'off'), "mesh file must be off"

    verts, faces = pcu.load_mesh_vf(path)

    return verts, faces

# 对点云进行不均匀的下采样
# 参数k会控制生成的点云最小间距，k越小，点云随机性越低，越均匀（有效最小值为1,此时变为均匀采样）
def random_downsample_points(points, num, k=0):
    # 先均匀下采样到一定倍数倍
    if k > 0 and points.shape[0] > num * k : 
        points = downsample_points(points, num * k)
    # 再随机选择一定数量的点进行输出
    out_point = points[np.random.choice(points.shape[0], num, replace=(num < points.shape[0])), :]
    return out_point

from utils.point_util import index_points, farthest_point_sample, knn_point
def groupKNN(pts, k, batch_size, random_sample=False):
    # 以seed_pt为中心选择k个点
    # pts: [N, 3]
    # seed_pt: [3]
    # random_sample : bool 选择是否随机采样/最远点采样
    # return: [b, k, 3]

    # numpy --> torch
    pts = torch.from_numpy(pts).unsqueeze(0)  # [1, N, 3]
    if not random_sample:
        # 最远点采样产生种子点:是不是代表固定就是某几个点
        fps_idx = farthest_point_sample(pts, batch_size) # [1, B]
        seed_pt = index_points(pts, fps_idx)    #[1, B, 3]
    else:
        # 随机产生种子点
        rand_idx = np.random.randint(0, pts.shape[1], size=(1,batch_size))   # [1, B]
        rand_idx = torch.from_numpy(rand_idx)   # [1, B]
        seed_pt = index_points(pts, rand_idx)    # [1, B, 3]
    

    idx = knn_point(k, pts, seed_pt)    # [1, B, k]
    grouped_xyz = index_points(pts, idx)    # [1, B, k, 3]
    grouped_xyz = grouped_xyz.squeeze(0) # [B, K, 3]

    return grouped_xyz.numpy()


# 将tensor类型的点云数据保存
def save_tensor_ply(xyz, path):
    # xyz: torch.tensor[N, 3]
    # path: str
    ply = xyz.cpu().detach().numpy()
    save_ply(ply, path)


# 将多个矩阵拼接到一起并保存，为了保持两者不干扰，两者之间有一个距离间隔
def save_merge_xyz(pts_list, dist, path):
    # pts_list: list of torch.tensor[N, 3]
    # distance: number
    # path: str
    merge_xyz = []
    centroid = torch.zeros((1, 3), dtype=torch.float32)
    for pts in pts_list:
        pts = pts + centroid.to(pts.device)
        merge_xyz.append(pts)
        centroid[0][0] = centroid[0][0] + dist

    merge_xyz = torch.cat(merge_xyz, dim=0)
    ply = merge_xyz.cpu().detach().numpy()
    save_ply(ply, path)

# 将两个点云通过颜色进行区别
def save_merge_xyz_color(xyz1, xyz2, path):
    # xyz1, xyz2: torch.tensor[N, 3]
    if not isinstance(xyz1, torch.Tensor):
        xyz1 = torch.from_numpy(xyz1)
    if not isinstance(xyz2, torch.Tensor):
        xyz2 = torch.from_numpy(xyz2)

    xyz2 = xyz2.to(xyz1.device)

    color1 = torch.zeros_like(xyz1)
    color2 = torch.zeros_like(xyz2)
    color1[:, 0] = 1    # red
    color2[:, 1] = 1    # green
    xyz = torch.cat([xyz1, xyz2], dim=0)
    colors = torch.cat([color1, color2], dim=0)
    xyz = xyz.cpu().detach().numpy()
    colors = colors.cpu().detach().numpy()
    save_ply(xyz, path, colors)

# 将两个点云通过颜色进行区别
def save_merge_xyz3_color(xyz1, xyz2, xyz3, path):
    # xyz1, xyz2, xyz3: torch.tensor[N, 3]
    if not isinstance(xyz1, torch.Tensor):
        xyz1 = torch.from_numpy(xyz1)
    if not isinstance(xyz2, torch.Tensor):
        xyz2 = torch.from_numpy(xyz2)
    if not isinstance(xyz3, torch.Tensor):
        xyz3 = torch.from_numpy(xyz3)

    xyz2 = xyz2.to(xyz1.device)
    xyz3 = xyz3.to(xyz1.device)

    color1 = torch.zeros_like(xyz1)
    color2 = torch.zeros_like(xyz2)
    color3 = torch.zeros_like(xyz3)
    color1[:, 0] = 1    # red
    color2[:, 1] = 1    # green
    color3[:, 2] = 1    # Blue
    xyz = torch.cat([xyz1, xyz2, xyz3], dim=0)
    colors = torch.cat([color1, color2, color3], dim=0)
    xyz = xyz.cpu().detach().numpy()
    colors = colors.cpu().detach().numpy()
    save_ply(xyz, path, colors)

# 将batch的点云面片通过红绿两个颜色进行显示
def save_merge_batch_xyz_color(pts_1, pts_2, path):
    # xyz1: [B, 3, N]
    # xyz2: [B, 3, M]
    pts_1 = pts_1.permute(0, 2, 1)
    pts_2 = pts_2.permute(0, 2, 1)
    merge_xyz = []
    merge_color = []
    centroid = torch.zeros((1, 3), dtype=torch.float32)
    for idx, (xyz1, xyz2) in enumerate(zip(pts_1, pts_2)):
        if idx >= 100:
            break

        bais_x = (idx%10)*3
        bais_y = (idx//10)*3
        centroid[0][0] = torch.tensor(bais_x, dtype=torch.float32)  #centroid[0][0] + bais_x    
        centroid[0][1] = torch.tensor(bais_y, dtype=torch.float32)  #centroid[0][1] + bais_y

        # 将两个点云混合
        color1 = torch.zeros_like(xyz1)
        color2 = torch.zeros_like(xyz2)
        color1[:, 0] = 1
        color2[:, 1] = 1
        xyz = torch.cat([xyz1, xyz2], dim=0)
        colors = torch.cat([color1, color2], dim=0)

        xyz = xyz + centroid.to(xyz.device)
        merge_xyz.append(xyz)
        
        merge_color.append(colors)
    
    merge_xyz = torch.cat(merge_xyz, dim=0)
    merge_color = torch.cat(merge_color, dim=0)

    merge_xyz = merge_xyz.cpu().detach().numpy()
    merge_color = merge_color.cpu().detach().numpy()
    save_ply(merge_xyz, path, merge_color)


def testPytorch3d():
    import pytorch3d
    import pytorch3d.loss
    import time

    # 读取点云数据并归一化
    
    xyz = read_ply("data/PUNet/pointclouds/train/10000_poisson/armadillo.ply")
    xyz, centroid, furthest_distance = normalize_point_cloud(xyz)

    ## 读取mesh数据并归一化
    t1 = time.time()
    verts, faces = load_off("data/PUNet/meshes/train/armadillo.off")
    t2 = time.time()
    verts, faces = load_off_own("data/PUNet/meshes/train/armadillo.off")
    t3 = time.time()
    print(t2-t1)
    print(t3-t2)
    # verts, _, _ = normalize_point_cloud(verts)
    verts = (verts - centroid) / furthest_distance
    verts = torch.FloatTensor(verts)
    faces = torch.LongTensor(faces)
    mesh = (verts, faces)

    ## 直接计算点云和mesh之间的距离
    def cal_dis(pts, verts, faces):
        # np.array[N, 3]   torch.tensor[M, 3] [Q, 3]
        pts = torch.from_numpy(pts).cuda(0) 
        verts = verts.cuda(0) 
        faces = faces.cuda(0) 
        pcls = pytorch3d.structures.Pointclouds([pts])
        meshes = pytorch3d.structures.Meshes([verts], [faces])
        dis =  pytorch3d.loss.point_mesh_face_distance(meshes, pcls)[1]

        return dis
    dis_1 = cal_dis(xyz, verts, faces)
    print("直接计算点云和mesh之间的距离: {}".format(dis_1.item()))
    save_merge_xyz_color(torch.from_numpy(xyz), verts, "output/visual/xyz_vert.ply")

    rotate_xyz = rotate_point_cloud_and_gt(np.expand_dims(xyz, 0).copy())[0][0]
    dis_1_1 = cal_dis(rotate_xyz, verts, faces)
    print("计算旋转点云和mesh之间的距离: {}".format(dis_1_1.item()))
    save_merge_xyz_color(torch.from_numpy(rotate_xyz), verts, "output/visual/rxyz_vert.ply")

    rotate_xyz, rotate_verts, _ = rotate_point_cloud_and_gt(np.expand_dims(xyz, 0).copy(), np.expand_dims(verts.numpy(), 0).copy())
    dis_1_2 = cal_dis(rotate_xyz[0], torch.from_numpy(rotate_verts[0]), faces)
    print("计算旋转点云和旋转mesh之间的距离: {}".format(dis_1_2.item()))
    save_merge_xyz_color(rotate_xyz[0], rotate_verts[0], "output/visual/rxyz_rvert.ply")

    ## 将点云选取一部分，计算和mesh之间的距离
    patch_num = 32
    patch_xyz = groupKNN(xyz, patch_num, 1, random_sample=False)[0]
    dis_2 = cal_dis(patch_xyz, verts, faces)
    print("将点云选取一部分，计算和mesh之间的距离: {}".format(dis_2.item()))
    save_merge_xyz_color(torch.from_numpy(patch_xyz), verts, "output/visual/patch_xyz_vert.ply")
    ## 将点云数据加噪，计算距离
    sigma = 0.005
    gaussian_xyz = jitter_perturbation_point_cloud_with_gaussian(np.expand_dims(xyz, 0), sigma=sigma, clip=1)
    gaussian_xyz = np.squeeze(gaussian_xyz, 0)
    dis_3 = cal_dis(gaussian_xyz, verts, faces)
    print("将点云数据加噪，计算和mesh之间的距离: {}".format(dis_3.item()))
    save_merge_xyz_color(torch.from_numpy(gaussian_xyz), verts, "output/visual/gaussian_xyz_vert.ply")
    # 将点云数据加噪并取一部分计算距离
    gaussian_patch_xyz = groupKNN(gaussian_xyz, patch_num, 1, random_sample=False)[0]
    dis_4 = cal_dis(gaussian_patch_xyz, verts, faces)
    print("将点云数据加噪并取一部分，计算和mesh之间的距离: {}".format(dis_4.item()))
    save_merge_xyz_color(torch.from_numpy(gaussian_patch_xyz), verts, "output/visual/gaussian_patch_xyz_vert.ply")
    # 将点云数据取一部分再加噪，计算和mesh之间的距离
    patch_gaussian_xyz = jitter_perturbation_point_cloud_with_gaussian(np.expand_dims(patch_xyz, 0), sigma=sigma*8, clip=1)
    patch_gaussian_xyz = np.squeeze(patch_gaussian_xyz, 0)
    dis_5 = cal_dis(patch_gaussian_xyz, verts, faces)
    print("将点云数据取一部分再加噪，计算和mesh之间的距离: {}".format(dis_5.item()))
    save_merge_xyz_color(torch.from_numpy(patch_gaussian_xyz), verts, "output/visual/patch_gaussian_xyz_vert.ply")


    return dis_5

def add_noise2xyz(xyz, noise_std):

    norm_xyz, centroid, furthest_distance =  normalize_point_cloud(xyz) # [B, N, 3]  [B, 1, 3]   [B, 1, 1]

    norm_xyz = torch.from_numpy(norm_xyz)
    noisy_xyz = norm_xyz + torch.randn_like(norm_xyz) * noise_std

    noisy_xyz = noisy_xyz * furthest_distance + centroid

    return noisy_xyz.numpy()

def test_add_noise():
    points = read_ply("data/PUNet/pointclouds/test/gt/camel_10000.ply")
    noisy_points = add_noise2xyz(points, 0.02)
    save_ply(noisy_points, "data/PUNet/pointclouds/test/gt/camel_10000n2e-2.ply")

if __name__ == "__main__":
    # testPytorch3d()   
    # test_add_noise()
    # points = read_ply("data/PUNet/pointclouds/test/input/input_simulated/kitten_noisy_scan0.1.xyz")[:, 0:3]
    # save_ply(points, "data/PUNet/pointclouds/test/input/input_simulated/kitten_noisy_scan0.1.ply")
    # save_merge_xyz([torch.from_numpy(points), torch.from_numpy(points)], 3, "output/visual/test_merge.ply")
    points = read_ply("data/PUNet/pointclouds/test/input/input_simulated_2/cow_noisy_scan0.1.ply")[:, 0:3]
    num = 10000
    down_points = downsample_points(points, num)
    save_ply(down_points, "data/PUNet/pointclouds/test/input/input_simulated_2/cow_10000n10e-3.ply")
    sigma = 0.015
    gaussian_point = jitter_perturbation_point_cloud_with_gaussian(np.expand_dims(down_points, 0), sigma=sigma, clip=1)
    gaussian_point = np.squeeze(gaussian_point, 0)
    save_ply(gaussian_point, "data/ball/model_1000_k4n15e-3.ply")

    gaussian_point = jitter_perturbation_point_cloud_with_gaussian(np.expand_dims(gaussian_point, 0), sigma=sigma, clip=1)
    gaussian_point = np.squeeze(gaussian_point, 0)
    save_ply(gaussian_point, "data/ball/model_1000_k4n15e-3n15e-3.ply")

    save_merge_xyz_color(torch.from_numpy(points), torch.from_numpy(gaussian_point), "output/visual/sample_noise_color.ply")
    down_gaussian_point = downsample_points(gaussian_point, num)
    save_merge_xyz_color(torch.from_numpy(down_points), torch.from_numpy(down_gaussian_point), "output/visual/down_sample_noise_color.ply")
    multi_jitter_gaussian_point = jitter_perturbation_point_cloud_with_multiplicative_noise(np.expand_dims(points, 0), sigma=0.15, clip=1)
    multi_jitter_gaussian_point = np.squeeze(multi_jitter_gaussian_point, 0)
    save_merge_xyz_color(torch.from_numpy(points), torch.from_numpy(multi_jitter_gaussian_point), "output/visual/sample_multi_noise_color.ply")



