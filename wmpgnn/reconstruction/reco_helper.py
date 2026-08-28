import pandas as pd
import numpy as np
import torch
from numpy import intersect1d

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from wmpgnn.reconstruction.signal_dict import particle_name


def lca_reco_matrix(graph, mode="reco"):
    edge_index = graph[('tracks', 'to', 'tracks')].edge_index.cpu()

    pd_matrix = pd.DataFrame(edge_index.T, columns=['senders', 'receivers'])
    if mode == "reco":
        edges = graph[('tracks', 'to', 'tracks')].lca.cpu()
        pd_matrix["LCA_dec"] = torch.argmax(edges, axis=-1).tolist()  # LCA decision
    else:
        pd_matrix["LCA_dec"] = graph[('tracks', 'to', 'tracks')].y.tolist()  # LCA decision
    pd_matrix.set_index(['senders', 'receivers'], inplace=True)
    pd_matrix = pd_matrix.reset_index()
    pd_matrix = pd_matrix[pd_matrix['senders'] < pd_matrix['receivers']]
    return pd_matrix


def get_final_keys(graph):
    """逐径迹键: 旧格式用 final_keys, 新格式(MC_normed)用 tracks.part_keys,
    公开LHCb碰撞数据无粒子键 -> 用 track 索引作为键 (保持键空间一致)"""
    if hasattr(graph, "final_keys"):
        return graph["final_keys"]
    if hasattr(graph["tracks"], "part_keys"):
        return graph["tracks"].part_keys
    return torch.arange(graph["tracks"].x.shape[0])


def get_truth_part_keys(graph):
    """信号末态粒子键: 旧格式用 truth_part_keys, 新格式用 tracks.sig_keys,
    公开LHCb碰撞数据无信号键 -> 用 ft==2 (b 径迹) 定义信号径迹"""
    if hasattr(graph, "truth_part_keys"):
        return graph["truth_part_keys"]
    if hasattr(graph["tracks"], "sig_keys"):
        return graph["tracks"].sig_keys
    ft = graph["tracks"].ft
    return torch.nonzero(ft == 2).flatten()


def get_truth_part_ids(graph):
    """信号末态粒子PDG: 旧格式用 truth_part_ids, 新格式用 tracks.sig_ids, 缺省填 0"""
    if hasattr(graph, "truth_part_ids"):
        return graph["truth_part_ids"]
    if hasattr(graph["tracks"], "sig_ids"):
        return graph["tracks"].sig_ids
    return torch.zeros(len(get_truth_part_keys(graph)), dtype=torch.long)


def lca_truth_matrix(graph):
    if hasattr(graph, "truth_senders"):
        # === 旧格式 (论文公开数据集 / CERN旧数据) ===
        senders = graph.truth_senders.cpu()
        receivers = graph.truth_receivers.cpu()

        truth_lca = pd.DataFrame(np.column_stack((senders, receivers)), columns=['senders', 'receivers'])
        truth_lca['LCA_dec'] = graph["truth_y"].cpu()
        truth_lca['LCA_id_label'] = list(map(particle_name, graph['truth_moth_ids'].cpu().numpy()))
        truth_lca['LCA_id'] = graph['truth_moth_ids'].cpu().numpy()
        truth_lca['TrueFullChainLCA'] = graph['lca_chain'].cpu()
        return truth_lca

    # === 新格式 (MC_normed) / 公开LHCb碰撞数据: 从 tt 边 y + 信号径迹键构建真值LCA矩阵 ===
    tt = graph[('tracks', 'to', 'tracks')]
    edge_index = tt.edge_index.cpu()
    y = tt.y.cpu()

    part_keys = get_final_keys(graph).cpu()
    sig_keys = get_truth_part_keys(graph).cpu()

    # 信号径迹 -> 其在 sig_keys 中的位置 (粒子键唯一, 用于重建粒子序列)
    pos_of_track = {}
    for sig_pos, sk in enumerate(sig_keys.tolist()):
        hit = (part_keys == sk).nonzero(as_tuple=False).flatten()
        if hit.numel() > 0:
            pos_of_track[int(hit[0])] = sig_pos

    # 收集 y>0 且两端都是信号径迹的 (去重: 只取 sender<receiver)
    rows = []
    for a, b, yab in zip(edge_index[0].tolist(), edge_index[1].tolist(), y.tolist()):
        if yab > 0 and a < b and a in pos_of_track and b in pos_of_track:
            rows.append((pos_of_track[a], pos_of_track[b], int(yab)))
    truth_lca = pd.DataFrame(rows, columns=['senders', 'receivers', 'LCA_dec'])
    if truth_lca.empty:
        return truth_lca
    # 去重 (公开LHCb碰撞数据 tt 边可能含重复对, 否则下游 multi-index 赋值会报错)
    truth_lca = truth_lca.drop_duplicates(subset=['senders', 'receivers'], keep='first')
    # 母粒子名: 新格式只存顶层B(不在LCA矩阵里逐对给出), 这里用符号代替(不影响聚类/分类)
    truth_lca['LCA_id_label'] = ""
    truth_lca['LCA_id'] = 0
    truth_lca['TrueFullChainLCA'] = truth_lca['LCA_dec'].astype(int)
    return truth_lca


def make_decay_dict(decay):
    decay_dict = {}
    for particle in decay:
        if particle not in decay_dict.keys():
            decay_dict[particle] = 1
        else:
            decay_dict[particle] += 1
    return decay_dict


def match_decays(decay1, decay2):
    decay_dict1 = make_decay_dict(decay1)
    decay_dict2 = make_decay_dict(decay2)
    if len(decay_dict1.keys()) != len(decay_dict2.keys()):
        return False
    decay_dict2_keys = decay_dict2.keys()
    for key in decay_dict1.keys():
        if key not in decay_dict2_keys:
            return False
        elif decay_dict1[key] != decay_dict2[key]:
            return False
    return True


def flatten(t):
    return [item for sublist in t for item in sublist]


def compute_LCA(anc1, anc2, max_depth):
    if (anc1 == []) or (anc2 == []):
        return 0

    common_ancestors = intersect1d(anc1, anc2).tolist()
    # IMPORTANT!!: the order of the ancestor indices reconstructed by this algorithm is the opposite of the one used in simulation, so the order must be reversed in this case.
    common_ancestors.reverse()

    if (common_ancestors == []):
        return 0

    lowest_common_ancestor = common_ancestors[-1]

    if (len(anc1) >= len(anc2)):
        max_length = anc1
    else:
        max_length = anc2
    lowest_common_ancestor_generation = max_length.index(
        lowest_common_ancestor)

    return max_depth - lowest_common_ancestor_generation


def reconstruct_decay(triang_LCA_matrix, particle_keys, ax=0, particle_ids=[], truth_level_simulation=0):
    num_clusters_per_order = {}
    for order_ in range(4):
        num_clusters_per_order[order_] = 0

    if particle_ids == []:
        labels = list(map(lambda x: 'k' + str(x), particle_keys))
    else:
        labels = list(map(lambda x, y: 'k' + str(x) + ':' +
                                       y, particle_keys, particle_ids))
    node_colors = []
    for l in labels:
        node_colors.append('#3e5948')

    max_full_chain_depth_in_event = -1

    # Create the global LCA matrix for the event, and remove null connections
    current_LCA_matrix = pd.DataFrame(triang_LCA_matrix, columns=[
        'senders', 'receivers', 'LCA_dec'])
    current_LCA_matrix = current_LCA_matrix[current_LCA_matrix['LCA_dec'] > 0]

    # Check against empty events
    if current_LCA_matrix.empty:
        return {}, num_clusters_per_order, max_full_chain_depth_in_event

    # Create a dictionary to store the true ID of the ancestors

    if truth_level_simulation:
        cluster_label_dict = pd.DataFrame(triang_LCA_matrix, columns=[
            'senders', 'receivers', 'LCA_id_label', 'TrueFullChainLCA'])
        cluster_label_dict.set_index(['senders', 'receivers'], inplace=True)
        max_full_chain_depth_in_event = max(
            cluster_label_dict['TrueFullChainLCA'].values)

    # Define an auxiliary matrix to later identify connected clusters
    num_nodes = len(particle_keys)

    clustering_adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # Get the maximum of the LCA matrix
    max_depth = np.max(current_LCA_matrix['LCA_dec'])

    adj_links_list = []

    composite_counter = num_nodes - 1

    for order in range(max_depth):

        LCA_matrix_subset = current_LCA_matrix[current_LCA_matrix['LCA_dec'] == 1]
        if LCA_matrix_subset.empty == False:

            # Reset the clustering adjacency matrix, and set it up to study the next LCA order
            clustering_adjacency_matrix = np.zeros(
                (composite_counter + 1, composite_counter + 1))

            for ie in range(LCA_matrix_subset.shape[0]):
                clustering_adjacency_matrix[LCA_matrix_subset.iloc[ie]
                ['senders']][LCA_matrix_subset.iloc[ie]['receivers']] = 1
                clustering_adjacency_matrix[LCA_matrix_subset.iloc[ie]
                ['receivers']][LCA_matrix_subset.iloc[ie]['senders']] = 1
            nx_graph = nx.from_numpy_array(clustering_adjacency_matrix)
            connected_components = [
                list(x) for x in nx.connected_components(nx_graph) if len(x) > 1]

            # Inspect the separate clusters iteratively
            cl_counter = 0
            for indices_in_cluster in connected_components:

                cl_counter += 1

                # Label the new cluster
                composite_counter += 1
                num_clusters_per_order[order] += 1
                if truth_level_simulation:
                    proxy_link = LCA_matrix_subset[(LCA_matrix_subset['senders'].isin(indices_in_cluster)) & (
                        LCA_matrix_subset['receivers'].isin(indices_in_cluster))].iloc[0]
                    labels.append('c' + str(composite_counter - num_nodes + 1) + ':' + cluster_label_dict.loc[(
                        proxy_link['senders'], proxy_link['receivers'])]['LCA_id_label'])
                else:
                    labels.append(
                        'reco_c' + str(composite_counter - num_nodes + 1))
                node_colors.append('#91b39d')

                # Pass the information to the reconstructed adjacency matrix
                for ind in indices_in_cluster:
                    new_df = pd.DataFrame({'senders': [ind],
                                           'receivers': [composite_counter],
                                           'link': [1]})
                    adj_links_list.append(new_df)

                # If there was any connection between the other nodes and any of the particles in the new cluster, connect those nodes to the new cluster as appropriate
                for sender in range(composite_counter):
                    if sender not in indices_in_cluster:
                        proxy_links = current_LCA_matrix[
                            ((current_LCA_matrix['senders'] == sender) & (current_LCA_matrix['receivers'].isin(
                                indices_in_cluster))) | ((current_LCA_matrix['senders'].isin(indices_in_cluster)) & (
                                    current_LCA_matrix['receivers'] == sender))]
                        if proxy_links.empty == False:
                            new_LCA_matrix_df = pd.DataFrame({
                                'senders': [sender],
                                'receivers': [composite_counter],
                                'LCA_dec': [max(proxy_links['LCA_dec'])]
                            })
                            LCA_matrix_list = []
                            LCA_matrix_list.append(current_LCA_matrix)
                            LCA_matrix_list.append(new_LCA_matrix_df)
                            current_LCA_matrix = pd.concat(
                                LCA_matrix_list, ignore_index=True)
                            if truth_level_simulation:
                                cluster_label_dict.loc[(sender, composite_counter), 'LCA_id_label'] = \
                                    cluster_label_dict.loc[(
                                        proxy_links['senders'].iloc[0],
                                        proxy_links['receivers'].iloc[0]), 'LCA_id_label']

                # Remove connections with the nodes inside the new cluster
                current_LCA_matrix = current_LCA_matrix[(current_LCA_matrix['senders'].isin(
                    indices_in_cluster) == False) & (current_LCA_matrix['receivers'].isin(indices_in_cluster) == False)]

        current_LCA_matrix['LCA_dec'] = current_LCA_matrix['LCA_dec'] - 1
        current_LCA_matrix = current_LCA_matrix[current_LCA_matrix['LCA_dec'] > 0]

    if (adj_links_list):
        adj_links = pd.concat(adj_links_list, ignore_index=True)

    # Plot the tree
    if ax != 0:
        G = nx.DiGraph()

        adj_senders = adj_links['senders'].to_list()
        adj_receivers = adj_links['receivers'].to_list()
        filtered_node_colors = []
        for i in range(len(labels)):
            if i in adj_senders or i in adj_receivers:
                G.add_node(labels[i])
                filtered_node_colors.append(node_colors[i])

        for ie in range(adj_links.shape[0]):
            edge = adj_links.iloc[ie]
            G.add_edge(labels[edge['receivers']], labels[edge['senders']])

        pos = graphviz_layout(G, prog='dot')
        nx.draw(G, pos, with_labels=False,
                node_color=filtered_node_colors, node_size=1300, ax=ax)
        label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
        nx.draw_networkx_labels(G, pos, font_size=14,
                                bbox=label_options, ax=ax)

    # Compute information per separated decay chain

    final_adjacency_matrix = np.zeros(
        (composite_counter + 1, composite_counter + 1))
    for ie in range(adj_links.shape[0]):
        final_adjacency_matrix[adj_links.iloc[ie]
        ['senders']][adj_links.iloc[ie]['receivers']] = 1
        final_adjacency_matrix[adj_links.iloc[ie]
        ['receivers']][adj_links.iloc[ie]['senders']] = 1
    nx_graph = nx.from_numpy_array(final_adjacency_matrix)
    connected_components = [
        list(x) for x in nx.connected_components(nx_graph) if len(x) > 1]

    clustered_keys = []
    clustered_concatenated_LCA_values = []

    for nodes_in_cluster in connected_components:

        # Identify the keys of the final particles in the cluster, and list them in ascending order
        index_from_key_dict = {}
        for node in nodes_in_cluster:
            if node < num_nodes:
                index_from_key_dict[particle_keys[node]] = node
        ordered_keys_in_cluster = list(index_from_key_dict.keys())
        ordered_keys_in_cluster.sort()
        clustered_keys.append(ordered_keys_in_cluster)

        # Identify the list of ancestors for each final state particle
        ancestor_lists_in_cluster = []
        for k in ordered_keys_in_cluster:
            node_index = index_from_key_dict[k]
            ancestor_list = []
            current_link = adj_links[adj_links['senders'] == node_index]
            while current_link.empty == False:
                current_receiver = current_link.iloc[0]['receivers']
                ancestor_list.append(current_receiver)
                current_link = adj_links[adj_links['senders']
                                         == current_receiver]
            ancestor_list.reverse()
            ancestor_lists_in_cluster.append(ancestor_list)
        max_decay_length = max([len(x) for x in ancestor_lists_in_cluster])

        # Compute the LCA values and concatenate them following a given order
        concatenated_LCA_values_in_cluster = []
        for in1 in range(len(ordered_keys_in_cluster)):
            for in2 in range(len(ordered_keys_in_cluster)):
                if in1 < in2:
                    concatenated_LCA_values_in_cluster.append(compute_LCA(
                        ancestor_lists_in_cluster[in1], ancestor_lists_in_cluster[in2], max_decay_length))
        clustered_concatenated_LCA_values.append(
            concatenated_LCA_values_in_cluster)

    # Store the cluster information in a dictionary, with entries given by the smallest key value
    cluster_dict = {}
    for ic in range(len(connected_components)):
        cluster_dict[clustered_keys[ic][0]] = {
            'node_keys': clustered_keys[ic], 'LCA_values': clustered_concatenated_LCA_values[ic],
            'labels': labels}

    return cluster_dict, num_clusters_per_order, max_full_chain_depth_in_event
