import numpy as np
import pandas as pd
import networkx as nx
import torch



# # ####################################################################

def calculate_degree_centrality(adjacency_matrix):
    num_nodes = len(adjacency_matrix)
    degree_centrality_matrix = np.zeros((num_nodes, num_nodes))

    np.fill_diagonal(adjacency_matrix, 1)

    for j in range(num_nodes):
        # Count the number of connections for each node j
        degree_centrality = np.sum(adjacency_matrix[j, :])

        # Store the degree centrality in the matrix
        degree_centrality_matrix[:, j] = degree_centrality  # Update entire column j

    # Divide each row i by the value of cell i, i
    for i in range(num_nodes):
        if degree_centrality_matrix[i, i] != 0:
              degree_centrality_matrix[i, :] /= degree_centrality_matrix[i, i]

    degree_centrality_matrix=np.where(degree_centrality_matrix < 1, 0, 1)             

    return degree_centrality_matrix

# # Read the adjacency matrix from the CSV file using Pandas
# adjacency_matrix = pd.read_csv("los_adj.csv", header=None).values
# print(type(adjacency_matrix))
# # Calculate degree centrality using the modified function
# degree_centrality_matrix = calculate_degree_centrality(adjacency_matrix)


# ####################################################################################




def calculate_closeness_centrality(adjacency_matrix):
    num_nodes = len(adjacency_matrix)
    closeness_centrality_matrix = np.zeros((num_nodes, num_nodes))

    # Make diagonal elements non-zero (replace zeros with ones)
    np.fill_diagonal(adjacency_matrix, 1)

    # Create a graph from the modified adjacency matrix
    graph = nx.from_numpy_array(adjacency_matrix)

    # Calculate closeness centrality
    closeness_centrality = nx.closeness_centrality(graph)

    for i in range(num_nodes):
        for j in range(num_nodes):
            # Store the value of cell j, j in cell i, j
            closeness_centrality_matrix[i, j] = closeness_centrality[j]

    # Divide each row i by the value of cell i, i if it's non-zero
    for i in range(num_nodes):
        if closeness_centrality_matrix[i, i] != 0:
            closeness_centrality_matrix[i, :] /= closeness_centrality_matrix[i, i]

    closeness_centrality_matrix=np.where(closeness_centrality_matrix < 1, 0, 1)           

    return closeness_centrality_matrix

# # Print the closeness centrality matrix
# print("Closeness Centrality Matrix:")
# print(closeness_centrality_matrix)


# # #####################################################################


def normalized_adj(A,is_sym=True, exponent=0.5):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave



def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    #adj = func(adj)
    adj=calculate_closeness_centrality(adj)
    adj=normalized_adj(adj)
    return adj


def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i : i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i : i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset
