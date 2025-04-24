import os
import pandas as pd
import pyvista as pv
import numpy as np
import torch
from torch.utils.data import Dataset

from train_dataloader import ProteinTrainDataset


class ProteinTestDataset(Dataset):
    def __init__(self, vtk_dir, csv_path, args):
        """
        Args:
            vtk_dir: 测试集VTK文件目录
            csv_path: 包含匿名ID的CSV文件路径
            args: 必须包含 num_point, use_normals, use_uniform_sample
        """
        self.vtk_dir = vtk_dir
        self.args = args
        df = pd.read_csv(csv_path)  
        self.anon_ids = df['anonymised_protein_id'].tolist()  # ["0.vtk", "1.vtk", ...]

    def __len__(self):
        return len(self.anon_ids)

    def __getitem__(self, idx):
        file = self.anon_ids[idx]
        mesh = pv.read(os.path.join(self.vtk_dir, file))

        
        points = mesh.points.astype(np.float32)
        potentials = mesh['Potential'].reshape(-1, 1).astype(np.float32)
        normal_pots = mesh['NormalPotential'].reshape(-1, 1).astype(np.float32)

        [x,y,z, potential, normal_potential]
        features = np.hstack([points, potentials, normal_pots])
        features[:, :3] = self._normalize_coords(features[:, :3])

        
        if self.args.use_uniform_sample:
            features = self.farthest_point_sample(features, self.args.num_point)
        else:
            features = features[:self.args.num_point]

        
        if self.args.use_normals:
            if 'Normals' in mesh.array_names:
                normals = mesh['Normals'][:self.args.num_point]
            else:
                normals = self._estimate_normals(features[:, :3])
            features = np.hstack([features, normals])

        return torch.from_numpy(features).float() 

    def _normalize_coords(self, coords):
        """归一化坐标到单位球内"""
        centroid = np.mean(coords, axis=0)
        coords -= centroid
        scale = np.max(np.sqrt(np.sum(coords ** 2, axis=1)))
        return coords / scale

    def farthest_point_sample(self, points, npoint):
        """最远点采样"""
        N, D = points.shape
        xyz = points[:, :3]  # 仅基于坐标采样
        centroids = np.zeros((npoint,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)

        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, axis=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)

        return points[centroids.astype(np.int32)]

    def _estimate_normals(self, points, k=32):
        """PCA估计法向量"""
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=k).fit(points)
        _, indices = knn.kneighbors(points)

        normals = np.zeros_like(points)
        for i in range(len(points)):
            neighbors = points[indices[i]]
            cov = np.cov(neighbors.T)
            _, _, v = np.linalg.svd(cov)
            normals[i] = v[2]  

        return normals

