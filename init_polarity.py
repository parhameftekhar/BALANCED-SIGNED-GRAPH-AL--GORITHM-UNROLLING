import os
import scipy.io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# 假设多个.mat文件路径存储在一个列表中
mat_files = []
for mat_file in os.listdir('./data/normal'):
    mat_files.append(os.path.join('./data/normal', mat_file))
for mat_file in os.listdir('./data/epilepsy'):
    mat_files.append(os.path.join('./data/epilepsy', mat_file))

# 存储所有计算得到的相关矩阵
correlation_matrices = []

# 循环处理每个.mat文件
for file in mat_files:
    # 读取.mat文件
    mat_data = scipy.io.loadmat(file)
    data = mat_data['data']
    
    # 提取数据
    for i in range(data.shape[0]):
        processed_data = np.array([_[1000:-500, 0] for _ in data[i, :-1]])
        correlation_matrix = np.corrcoef(processed_data)
        if np.any(np.isnan(correlation_matrix)) or np.any(np.isinf(correlation_matrix)):
            print(f"Skipping file {file} (index {i}) due to NaN or inf in correlation matrix")
            continue  # 跳过此文件和i
        correlation_matrices.append(correlation_matrix)

# 计算所有相关矩阵的平均值
mean_correlation_matrix = np.mean(correlation_matrices, axis=0)

# 层次聚类并获取聚类标签
linkage_matrix = linkage(1 - mean_correlation_matrix, method='average')
dendro = dendrogram(linkage_matrix, no_plot=True)
order = dendro['leaves']
cluster_labels = np.zeros(mean_correlation_matrix.shape[0])
for i, idx in enumerate(order[:mean_correlation_matrix.shape[0] // 2]):
    cluster_labels[idx] = 0  # 第一部分
for i, idx in enumerate(order[mean_correlation_matrix.shape[0] // 2:]):
    cluster_labels[idx] = 1  # 第二部分

# 创建热力图
reordered_matrix = mean_correlation_matrix[order, :][:, order]
plt.figure(figsize=(10, 8))
sns.heatmap(reordered_matrix, annot=False, cmap="RdBu_r", 
            xticklabels=[f'Vector {i+1}' for i in range(mean_correlation_matrix.shape[0])], 
            yticklabels=[f'Vector {i+1}' for i in range(mean_correlation_matrix.shape[0])], 
            cbar_kws={'label': 'Correlation Coefficient'}, vmin=-1, vmax=1)
plt.title("Average Correlation Matrix Heatmap with Clustering")
plt.savefig('./fig/average_correlation_matrix_heatmap.png')

# 只保存 cluster_labels 到 MAT 文件
print(cluster_labels)
scipy.io.savemat('./results/cluster_labels.mat', {'cluster_labels': cluster_labels})
