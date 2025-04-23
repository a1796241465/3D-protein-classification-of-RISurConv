import os
import numpy as np
import torch
import pandas as pd
import pyvista as pv
from torch.utils.data import Dataset
from tqdm import tqdm
import re


class ProteinTrainDataset(Dataset):
    def __init__(self, vtk_dir, csv_path, args):
        """
        Args:
            vtk_dir: 包含VTK文件的目录（如 `train_vtk/`）
            csv_path: 包含 `protein_id` 和 `class_id` 的CSV文件路径
            args: 必须包含 num_point, use_normals, use_uniform_sample
        """
        self.vtk_dir = vtk_dir
        self.args = args

        # 从CSV加载标签映射 {protein_id: class_id}
        df = pd.read_csv(csv_path)
        self.label_map = dict(zip(df['protein_id'], df['class_id']))

        # 获取所有VTK文件并匹配标签
        self.files = []
        for file in os.listdir(vtk_dir):
            if file.endswith('.vtk'):
                protein_id = file.replace('.vtk', '')  # 获取文件名，去掉扩展名

                # 分割文件名各部分
                parts = re.split(r'[_:]', protein_id)
                if len(parts) >= 0:  # 确保有足够的部分
                    # 重组文件名：第一部分_第二部分:第三部分:第四部分_第五部分...
                    protein_id_csv_format = f"{parts[0]}_{parts[1]}:{parts[2]}:{parts[3]}_{'_'.join(parts[4:])}"
                else:
                    print(f"Warning: Unexpected filename format {protein_id}, skipping")
                    continue

                
                if protein_id_csv_format in self.label_map:
                    self.files.append((file, self.label_map[protein_id_csv_format]))
                else:
                    print(f"Warning: {protein_id_csv_format} not found in CSV, skipping file {file}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file, label = self.files[idx]
        mesh = pv.read(os.path.join(self.vtk_dir, file))

        # 提取点云和属性
        points = mesh.points.astype(np.float32)  # [N, 3]

        potentials = mesh['Potential'].reshape(-1, 1).astype(np.float32)  # [N, 1]
        normal_pots = mesh['NormalPotential'].reshape(-1, 1).astype(np.float32)  # [N, 1]

        # 拼接特征 [x,y,z, potential, normal_potential] -> [N, 5]
        features = np.hstack([points, potentials, normal_pots])
        features[:, :3] = self._normalize_coords(features[:, :3])  

        # 降采样
        if self.args.use_uniform_sample:
            features = self.farthest_point_sample(features, self.args.num_point)
        else:
            features = features[:self.args.num_point]

        # 添加法向量
        if self.args.use_normals:
            if 'normals' in mesh.array_names:
                normals = mesh['Normals'][:self.args.num_point]  
            else:
                # PCA估计法向量
                normals = self._estimate_normals(features[:, :3])
            features = np.hstack([features, normals])  # [N, 8]

        return torch.from_numpy(features).float(), torch.tensor(label)

    def _normalize_coords(self, coords):
        """归一化坐标到单位球内"""
        centroid = np.mean(coords, axis=0)
        coords -= centroid
        scale = np.max(np.sqrt(np.sum(coords ** 2, axis=1)))
        return coords / scale

    def farthest_point_sample(self, points, npoint):
        """
        最远点采样 (FPS)
        Args:
            points: [N, D] 输入点云（仅使用前3维坐标进行采样）
            npoint: 目标点数
        Returns:
            sampled_points: [npoint, D] 采样后的点云
        """
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

        sampled_points = points[centroids.astype(np.int32)]
        return sampled_points

    def _estimate_normals(self, points, k=32):
        """
        通过PCA估计法向量
        Args:
            points: [N, 3] 点云坐标
            k: KNN邻居数
        Returns:
            normals: [N, 3] 估计的法向量
        """
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=k).fit(points)
        _, indices = knn.kneighbors(points)

        normals = np.zeros_like(points)
        for i in range(len(points)):
            neighbors = points[indices[i]]
            cov = np.cov(neighbors.T)
            _, _, v = np.linalg.svd(cov)
            normals[i] = v[2]  # 最小特征值对应的向量

        return normals
