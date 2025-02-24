import scipy.io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# 读取.mat文件
mat_data = scipy.io.loadmat('./data/normal/2_Normal (11).mat')  # 替换为你的文件路径
mat_data = scipy.io.loadmat('./data/epilepsy/1_epilepsy (12).mat')  # 替换为你的文件路径
# print(mat_data.keys())
data = mat_data['data']
print(f"Data shape: {data.shape}")

def plot_origin(data):
    num_plots = data.shape[0]
    cols = 1  # 每行1个子图
    rows = int(np.ceil(num_plots / cols))  # 根据数据的数量自动计算行数
    plt.figure(figsize=(cols * 20, rows * 5))
    for i, d in enumerate(data):
        plt.subplot(rows, cols, i + 1)  # 行列索引
        plt.plot(d)
        plt.title(f"Plot {i + 1}")
    plt.tight_layout()
    plt.savefig('./fig/combined_plot.png')
    plt.close()

def plot_correlation(data):
    # 计算相关系数矩阵
    correlation_matrix = np.corrcoef(data)
    linkage_matrix = linkage(1 - correlation_matrix, method='average')  # 1 - 相关系数计算距离矩阵
    dendro = dendrogram(linkage_matrix, no_plot=True)  # 获取聚类的顺序
    order = dendro['leaves']
    sorted_correlation_matrix = correlation_matrix[order, :][:, order]

    # 创建热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_correlation_matrix, annot=False, cmap="RdBu_r",  # 不显示单元格的数值
                xticklabels=[f'Vector {i+1}' for i in order], 
                yticklabels=[f'Vector {i+1}' for i in order], 
                cbar_kws={'label': 'Correlation Coefficient'}, vmin=-1, vmax=1)
    plt.title("Sorted Correlation Matrix Heatmap (Clustered into 2)")
    plt.show()
    

data = np.array([_[1000:, 0] for _ in data[3, :-1]])
plot_correlation(data)