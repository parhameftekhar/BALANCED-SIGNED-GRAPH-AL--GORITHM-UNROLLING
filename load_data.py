import torch
import numpy as np
import scipy, os
from torch_geometric.data import Data

channels = [
    (0, 'FP1', 'A1'),
    (1, 'FP2', 'A2'),
    (2, 'F3', 'A1'),
    (3, 'F4', 'A2'),
    (4, 'FZ', 'A2'),
    (5, 'C3', 'A1'),
    (6, 'C4', 'A2'),
    (7, 'CZ', 'A1'),
    (8, 'P3', 'A1'),
    (9, 'P4', 'A2'),
    (10, 'PZ', 'A2'),
    (11, 'O1', 'A1'),
    (12, 'O2', 'A2'),
    (13, 'F7', 'A1'),
    (14, 'F8', 'A1'),
    (15, 'T3', 'A1'),
    (16, 'T4', 'A2'),
    (17, 'T5', 'A1'),
    (18, 'FP1', 'A1'),
    (19, 'FP2', 'A2'),
    (20, 'F3', 'A1'),
    (21, 'F4', 'A2'),
    (22, 'FZ', 'A2'),
    (23, 'C3', 'A1'),
    (24, 'C4', 'A2'),
    (25, 'CZ', 'A1'),
    (26, 'P3', 'A1'),
    (27, 'P4', 'A2'),
    (28, 'PZ', 'A2'),
    (29, 'O1', 'A1'),
    (30, 'O2', 'A2'),
    (31, 'F7', 'A1'),
    (32, 'F8', 'A1'),
    (33, 'T3', 'A1'),
    (34, 'T4', 'A2')
]

def cal_edge_index(graph_seq):
    # 基图，对于有公共节点的电极则建立边
    electrode_dict = {}
    for idx, elec_1, elec_2 in channels:
        if elec_1 not in electrode_dict:
            electrode_dict[elec_1] = []
        else:
            electrode_dict[elec_1].append(idx)
        
        if elec_2 not in electrode_dict:
            electrode_dict[elec_2] = []
        else:
            electrode_dict[elec_2].append(idx)
    edges = []
    for elec, nodes in electrode_dict.items():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                edges.append((nodes[i], nodes[j]))
                
    # 按照graph_seq长度搭建完整的图
    all_edges = edges.copy()
    for g in range(1, graph_seq):
        for e in edges:
            all_edges.append((e[0]+g*len(channels), e[1]+g*len(channels)))
    for g in range(1,graph_seq):
        for c in range(len(channels)):
            all_edges.append((c+(g-1)*len(channels),c+g*len(channels)))
    # print(all_edges)
    print(len(all_edges))
    return torch.tensor(all_edges, dtype=torch.long).t().contiguous()

def normalize_data(data_list):
    sum_x, sum_x2, total_count = 0.0, 0.0, 0

    # 逐步计算均值和方差
    for data in data_list:
        sum_x += data.x.sum()
        sum_x2 += (data.x ** 2).sum()
        total_count += data.x.numel()

    mean = sum_x / total_count
    std = torch.sqrt(sum_x2 / total_count - mean**2) + 1e-6

    for data in data_list:
        data.x = (data.x - mean) / std  # 标准化
    return data_list

def load_graph_data(data_dir='./data', graph_seq=6, signal_len=1000):
    # 获取图结构
    print("Constructing Graph Structure...")
    edge_index = cal_edge_index(graph_seq)
    
    # 读入节点特征
    print("Loading Nodes Features...")
    all_data = []
    normal_data_dir = os.path.join(data_dir, 'normal')
    for mat_file in os.listdir(normal_data_dir):
        mat_data = scipy.io.loadmat(os.path.join(normal_data_dir, mat_file))['data'] # (n_sample, 36) x (7500, 1)
        for sample in mat_data:
            signals = np.array([_[1000:-500,0] for _ in sample[:-1]])
            x = None
            for g in range(graph_seq):
                if x is None:
                    x = torch.tensor(signals[:,signal_len*g:signal_len*(g+1)])
                else:
                    x = torch.cat((x, torch.tensor(signals[:,signal_len*g:signal_len*(g+1)])))
            y = torch.tensor([0], dtype=torch.long)
            all_data.append(Data(x=x, edge_index=edge_index, y=y))
    epilepsy_data_dir = os.path.join(data_dir, 'epilepsy')
    for mat_file in os.listdir(epilepsy_data_dir):
        mat_data = scipy.io.loadmat(os.path.join(epilepsy_data_dir, mat_file))['data'] # (n_sample, 36) x (7500, 1)
        for sample in mat_data:
            signals = np.array([_[1000:-500,0] for _ in sample[:-1]])
            x = None
            for g in range(graph_seq):
                if x is None:
                    x = torch.tensor(signals[:,signal_len*g:signal_len*(g+1)])
                else:
                    x = torch.cat((x, torch.tensor(signals[:,signal_len*g:signal_len*(g+1)])))
            y = torch.tensor([0], dtype=torch.long)
            all_data.append(Data(x=x, edge_index=edge_index, y=y))
    print(len(all_data),all_data[0])
    return normalize_data(all_data)
            
if __name__ == "__main__":
    load_graph_data()