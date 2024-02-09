import os
import numpy as np
from scipy.sparse import load_npz, save_npz, csr_matrix, lil_matrix
from scipy.sparse.csgraph import shortest_path
import heapq
import torch
from scipy.spatial.distance import cdist
from scipy.sparse import lil_matrix, csr_matrix


def read_and_combine_matrices(data_root):
    # 读取矩阵文件
    train_matrix = load_npz(os.path.join(data_root, 'train_matrix.npz'))
    # print(train_matrix.shape)
    val_matrix = load_npz(os.path.join(data_root, 'val_matrix.npz'))
    # print(val_matrix.shape)
    test_matrix = load_npz(os.path.join(data_root, 'test_matrix.npz'))
    # print(test_matrix.shape)

    # 合并矩阵
    combined_matrix = train_matrix + val_matrix + test_matrix

    # 确保合并后的矩阵仍然是二元的（1 表示交互，0 表示无交互）
    combined_matrix[combined_matrix > 1] = 1

    return combined_matrix


# def create_networkx_graph_from_sparse_matrix(sparse_matrix):
#     G = nx.Graph()
#     cx = sparse_matrix.tocoo()
#     for i, j, v in zip(cx.row, cx.col, cx.data):
#         if v > 0:
#             G.add_edge(i, j, weight=v)
#     return G

# def detect_communities(graph):
#     # 使用 Girvan-Newman 算法，这会返回一个社区生成器
#     graph = create_networkx_graph_from_sparse_matrix(graph)
#
#     communities_generator = community.girvan_newman(graph)
#     # 选择第一次划分的社区，这将是最高级别的划分
#     top_level_communities = next(communities_generator)
#     # 将社区结果转换为列表
#     return [list(community) for community in top_level_communities]


# def create_community_interaction_matrix(graph, num_nodes):
#     # 检测社区
#     communities = detect_communities(graph)
#
#     # 构建节点到社区的映射
#     node_to_community = {}
#     for i, community in enumerate(communities):
#         for node in community:
#             node_to_community[node] = i
#
#     # 初始化交互矩阵
#     interaction_matrix = lil_matrix((num_nodes, num_nodes), dtype=np.float32)
#
#     # 根据节点是否在同一社区内设置 interaction 值
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if i == j:
#                 # 自身节点，不计算
#                 continue
#             interaction = 1 if node_to_community.get(i) == node_to_community.get(j) else 0
#             interaction_matrix[i, j] = interaction
#
#     return interaction_matrix

# def generate_embeddings(graph, dimensions=64, walk_length=30, num_walks=200, workers=4):
#     # 初始化 Node2Vec 模型
#     node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length,
#                         num_walks=num_walks, workers=workers)
#     # 训练模型
#     model = node2vec.fit(window=10, min_count=1, batch_words=4)
#     return model

# def create_interaction_matrix_from_embeddings(embeddings, num_nodes):
#     # 使用 embeddings 计算节点间的相似度
#     similarity_matrix = cosine_similarity(embeddings)
#
#     # 初始化 interaction_matrix 为 lil_matrix
#     interaction_matrix = lil_matrix((num_nodes, num_nodes), dtype=np.float32)
#
#     # 填充 interaction_matrix
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             # 你可以根据相似度直接设置 interaction 值，或者设置一个阈值
#             interaction_matrix[i, j] = similarity_matrix[i, j]
#
#     return interaction_matrix.tocsr()

def create_interaction_matrix(combined_matrix):
    num_users, num_items = combined_matrix.shape
    # total_size = num_users + num_items
    num_nodes = num_users + num_items
    # graph = np.zeros((num_nodes, num_nodes))
    graph = lil_matrix((num_nodes, num_nodes))
    if not isinstance(combined_matrix, lil_matrix):
        combined_matrix = combined_matrix.tolil()
    # 填充用户-物品交互，构建graph
    graph[:num_users, num_users:] = combined_matrix
    graph[num_users:, :num_users] = combined_matrix.T

    # 计算graph矩阵的node之间的最小连通距离，构建dist_matrix
    dist_matrix = shortest_path(csgraph=graph, directed=False, method='FW')
    transformed_dist_matrix = 6 - dist_matrix
    transformed_dist_matrix = np.maximum(transformed_dist_matrix, 0)
    min_val = np.min(transformed_dist_matrix)
    max_val = np.max(transformed_dist_matrix)
    dist_matrix = (transformed_dist_matrix - min_val) / (max_val - min_val)
    print('dist_matrix', dist_matrix)
    print('dist_matrix.shape', dist_matrix.shape)

    # 计算聚类，分析类别
    # cluster_matrix = create_community_interaction_matrix(graph, num_nodes)
    # min_val = np.min(cluster_matrix)
    # max_val = np.max(cluster_matrix)
    # scaled_matrix = (cluster_matrix - min_val) / (max_val - min_val)
    # cluster_matrix = scaled_matrix
    # print('cluster_matrix', cluster_matrix)
    # print('cluster_matrix.shape', cluster_matrix.shape)

    # embedding_matrix
    # embeddings_model = generate_embeddings(graph)
    # embeddings = embeddings_model.wv.vectors
    # embedding_matrix = create_interaction_matrix_from_embeddings(embeddings, len(embeddings_model.wv))
    # min_val = np.min(embedding_matrix)
    # max_val = np.max(embedding_matrix)
    # scaled_matrix = (embedding_matrix - min_val) / (max_val - min_val)
    # embedding_matrix = scaled_matrix
    # print('embedding_matrix', embedding_matrix)
    # print('embedding_matrix.shape', embedding_matrix.shape)

    # graph + dist_matrix + cluster_matrix + embedding_matrix
    # 放缩到0,1之间
    graph_dense = graph.toarray() if isinstance(graph, lil_matrix) else graph
    dist_matrix_dense = dist_matrix.toarray() if isinstance(dist_matrix, lil_matrix) else dist_matrix
    # cluster_matrix_dense = cluster_matrix.toarray() if isinstance(cluster_matrix, lil_matrix) else cluster_matrix
    # embedding_matrix_dense = embedding_matrix.toarray() if isinstance(embedding_matrix,
    #                                                                   lil_matrix) else embedding_matrix
    # total_matrix = graph_dense + dist_matrix_dense + cluster_matrix_dense + embedding_matrix_dense
    total_matrix = graph_dense + dist_matrix_dense
    # 应用 Min-Max 标准化
    min_val = np.min(total_matrix)
    max_val = np.max(total_matrix)
    scaled_matrix = (total_matrix - min_val) / (max_val - min_val)

    # 将缩放后的矩阵转换回稀疏格式
    interaction_matrix = csr_matrix(scaled_matrix)

    return interaction_matrix


def save_graph_matrix(combined_matrix, data_root):
    num_users, num_items = combined_matrix.shape
    num_nodes = num_users + num_items
    graph = lil_matrix((num_nodes, num_nodes))
    if not isinstance(combined_matrix, lil_matrix):
        combined_matrix = combined_matrix.tolil()

    graph[:num_users, num_users:] = combined_matrix
    graph[num_users:, :num_users] = combined_matrix.T

    file_path = os.path.join(data_root, 'graph_matrix.npz')
    save_npz(file_path, graph.tocsr())

    return graph


def dijkstra(graph, start):
    """
    Dijkstra's algorithm for shortest paths
    :param graph: Graph represented as a weight matrix (numpy array)
    :param start: Starting node
    :return: Distance matrix from start to every other node
    """
    num_nodes = graph.shape[0]
    visited = [False] * num_nodes
    dist = [float('inf')] * num_nodes
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        current_dist, current_vertex = heapq.heappop(pq)
        visited[current_vertex] = True

        for neighbor in range(num_nodes):
            if graph[current_vertex, neighbor] > 0 and not visited[neighbor]:
                distance = current_dist + graph[current_vertex, neighbor]
                if distance < dist[neighbor]:
                    dist[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))

    return dist


def dijkstra_all_pairs(graph):
    """
    Apply Dijkstra's algorithm for all pairs in the graph
    :param graph: Graph represented as a weight matrix (numpy array)
    :return: Matrix of shortest path distances between all pairs of nodes
    """
    num_nodes = graph.shape[0]
    all_pairs_shortest_path = np.zeros((num_nodes, num_nodes))

    for node in range(num_nodes):
        print(node)
        all_pairs_shortest_path[node] = dijkstra(graph, node)

    return all_pairs_shortest_path


def save_and_normalize_dist_matrix(graph, data_root):
    # dist_matrix = shortest_path(csgraph=graph, directed=False, method='FW')
    graph_array = graph.toarray()

    # Compute all pairs shortest paths
    dist_matrix = dijkstra_all_pairs(graph_array)
    # dist_matrix = floyd_warshall(graph)
    print(dist_matrix)
    transformed_dist_matrix = 6 - dist_matrix
    transformed_dist_matrix = np.maximum(transformed_dist_matrix, 0)
    min_val = np.min(transformed_dist_matrix)
    max_val = np.max(transformed_dist_matrix)
    normalized_dist_matrix = (transformed_dist_matrix - min_val) / (max_val - min_val)
    print(normalized_dist_matrix.shape)

    file_path = os.path.join(data_root, 'dist_matrix.npz')
    np.savez(file_path, normalized_dist_matrix)

    # return normalized_dist_matrix


def combine_and_save_interaction_matrix(data_root):
    graph_matrix = load_npz(os.path.join(data_root, 'graph_matrix.npz'))
    dist_matrix = np.load(os.path.join(data_root, 'dist_matrix.npz'))
    # print(dist_matrix['arr_0'])
    # print('dist_matrix.shape',dist_matrix.shape)
    # print('dist_matrix',dist_matrix)
    # print(dist_matrix.files)

    graph_dense = graph_matrix.toarray()
    # print('graph_dense',graph_dense)
    # print(np.max(graph_dense))
    dist_matrix_dense = dist_matrix['arr_0']

    total_matrix = graph_dense + dist_matrix_dense
    min_val = np.min(total_matrix)
    max_val = np.max(total_matrix)
    scaled_matrix = (total_matrix - min_val) / (max_val - min_val)

    interaction_matrix = scaled_matrix
    print(interaction_matrix.shape)

    # file_path = os.path.join(data_root, 'interaction_matrix.npz')
    # np.savez(file_path, interaction_matrix)


def save_interaction_matrix(data_root):
    graph_matrix = load_npz(os.path.join(data_root, 'graph_matrix.npz'))

    # Convert to LIL format for efficient element-wise operations
    graph_matrix = graph_matrix.tolil()

    # Set diagonal elements to 1
    graph_matrix.setdiag(1)
    print('graph_matrix.setdiag(1)')

    # Set off-diagonal zeros to 0.5
    graph_matrix[graph_matrix == 0] = 0.5
    print('000')
    print('graph_matrix[graph_matrix == 0] = 0.5')

    # Convert back to CSR format (or keep it in LIL if that's better for subsequent operations)
    interaction_matrix = graph_matrix.tocsr()

    print(interaction_matrix.shape)

    # Save the interaction_matrix
    # Make sure to adjust the path and format according to your needs
    save_npz(os.path.join(data_root, 'interaction_matrix.npz'), interaction_matrix)

    # file_path = os.path.join(data_root, 'interaction_matrix.npz')
    # np.savez(file_path, interaction_matrix)


def create_interaction_matrix_gpu(combined_matrix):
    num_users, num_items = combined_matrix.shape
    num_nodes = num_users + num_items
    embedding_dim = 64  # 嵌入向量的维度

    # 初始化用户和物品的嵌入矩阵
    user_embeddings = torch.randn(num_users, embedding_dim, device='cuda', requires_grad=True)
    item_embeddings = torch.randn(num_items, embedding_dim, device='cuda', requires_grad=True)

    # 将combined_matrix转换为稠密格式并移动到GPU
    interaction_matrix_dense = torch.tensor(combined_matrix.todense(), device='cuda', dtype=torch.float32)

    optimizer = torch.optim.Adam([user_embeddings, item_embeddings], lr=0.01)

    # 简化的训练过程
    for epoch in range(100):
        optimizer.zero_grad()

        # 基于当前嵌入计算用户-物品交互矩阵的预测
        predicted_matrix = torch.matmul(user_embeddings, item_embeddings.T)

        # 计算损失（例如，均方误差）
        loss = torch.nn.functional.mse_loss(predicted_matrix, interaction_matrix_dense)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # 将嵌入向量拼接，形成节点嵌入
    all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0).cpu().detach().numpy()

    # 计算节点间的欧氏距离
    dist_matrix = cdist(all_embeddings, all_embeddings, metric='euclidean')

    # 距离的逆转换为相似度
    dist_matrix = 1 / (1 + dist_matrix)
    min_val, max_val = np.min(dist_matrix), np.max(dist_matrix)
    normalized_dist_matrix = (dist_matrix - min_val) / (max_val - min_val)

    # 返回归一化的相似度矩阵作为交互矩阵
    interaction_matrix = csr_matrix(normalized_dist_matrix)

    return interaction_matrix

# 假设您的数据根目录为以下路径
# dataset = 'beauty'
# dataset = 'sports'
# dataset = 'toys'
# dataset = 'yelp'
# dataset = 'scientific'
# dataset = 'arts'
# dataset = 'instruments'
# dataset = 'office'
# dataset = 'pantry'
# dataset = 'luxury'
# dataset = 'music'
dataset = 'garden'
# dataset = 'food'
server_root = "/home/local/ASURITE/xwang735/LLM4REC/LLM4Rec"
# server_root = "/home/wxy/LLM4Rec"
gpt2_server_root = server_root
data_root = os.path.join(gpt2_server_root, "dataset", dataset)

combined_matrix = read_and_combine_matrices(data_root)

# 如果需要，可以查看合并后的矩阵维度
print(combined_matrix.shape)
interaction_matrix = create_interaction_matrix(combined_matrix)
# interaction_matrix = create_interaction_matrix_gpu(combined_matrix)
print(interaction_matrix.shape)
file_path = os.path.join(data_root, 'interaction_matrix.npz')
save_npz(file_path, interaction_matrix)

# combined_matrix = read_and_combine_matrices(data_root)
# print('combined_matrix.shape', combined_matrix.shape)
# graph = save_graph_matrix(combined_matrix, data_root)
# print('graph.shape', graph.shape)
# save_and_normalize_dist_matrix(graph, data_root)
# combine_and_save_interaction_matrix(data_root)
# save_interaction_matrix(data_root)
print('done')
