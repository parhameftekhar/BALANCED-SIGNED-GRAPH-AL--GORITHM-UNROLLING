import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data
import torch.nn.functional as F
from load_data import load_graph_data

# CNN特征提取器
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=-1)  # 平均池化
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 训练 M，确保 PSD
class GraphDenoisingModel(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, num_nodes, beta):
        super().__init__()
        self.cnn = CNNFeatureExtractor(input_dim, feature_dim)  # CNN 提取特征
        self.Q = torch.nn.Parameter(torch.randn(feature_dim, feature_dim))  # 训练 Q 确保 M 是 PSD
        self.beta = beta.to(torch.float32)  # 初始化所有极性为 1
        self.num_nodes = num_nodes
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def compute_M(self):
        """ M = Q Q^T 确保 PSD """
        return torch.matmul(self.Q, self.Q.T)

    def compute_distance(self, f):
        """ 计算 d_ij = (f_i - f_j)^T M (f_i - f_j) """
        M = self.compute_M()
        diff = f.unsqueeze(1) - f.unsqueeze(0)  # (N, N, d)
        dists = torch.einsum('bnd,dd,bnd->bn', diff, M, diff)  # 计算 (fi - fj)^T M (fi - fj)
        return dists

    def compute_weights(self, dists):
        """ 计算边权重 w_{i,j} = β_i β_j exp(-d_ij) """
        weights = torch.exp(-dists)  # 计算 exp(-d_ij)
        signs = self.beta.unsqueeze(1) * self.beta.unsqueeze(0)  # 计算 β_i * β_j
        return signs * weights  # 保证符号符合平衡图

    def forward(self, data):
        x = self.cnn(data.x)  # CNN 提取特征 f_i
        dists = self.compute_distance(x)  # 计算距离 d_{i,j}
        weights = self.compute_weights(dists)  # 计算 w_{i,j}
        
        # 计算拉普拉斯矩阵 Lb
        degree = torch.diag(weights.sum(dim=1))
        Lb = degree - weights
        
        return Lb, x

    def optimize_beta(self, data, Lb):
        """ 迭代优化 β """
        num_nodes = self.num_nodes
        edge_index = data.edge_index
        update_flag = True
        while update_flag:
            update_flag = False
            for i in range(num_nodes):
                connected_nodes = edge_index[1][edge_index[0] == i]
                if len(connected_nodes) == 0:
                    continue
                
                origin_loss = torch.matmul(data.x.T, torch.matmul(Lb, data.x)).mean()
                self.beta[i] *= -1
                updated_Lb, _ = self.forward(data)
                updated_loss = torch.matmul(data.x.T, torch.matmul(updated_Lb, data.x)).mean()
                if updated_loss > origin_loss:
                    self.beta[i] *= -1
                else:
                    update_flag = True

    def train_step(self, data):
        """ 训练 CNN + M，优化 β """
        self.optimizer.zero_grad()
        Lb, x_pred = self.forward(data)

        # 计算损失
        loss_denoising = F.mse_loss(x_pred, data.x)
        # loss_GLR = torch.matmul(data.x, torch.matmul(Lb, data.x)).mean()
        
        loss = loss_denoising # + 0.1 * loss_GLR
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新 CNN 和 M

        # 更新 β（非梯度下降）
        self.optimize_beta(data, Lb)
        return loss.item()

all_data = load_graph_data()
# 训练模型
num_nodes = all_data[0].x.shape[0]
input_dim = all_data[0].x.shape[1]
model = GraphDenoisingModel(input_dim, feature_dim=64, num_nodes=num_nodes)
for epoch in range(100):
    loss = 0.0
    for data in all_data:
        loss += model.train_step(data) / len(all_data)
    print(f"Epoch {epoch}, Loss: {loss}")
