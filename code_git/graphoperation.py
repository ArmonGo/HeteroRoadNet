
import osmnx as ox
import numpy as np
from shapely import LineString
from rustworkx import steiner_tree
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils.map import map_index
import torch 
import rustworkx as rx 
from sklearn.neighbors import KDTree
import copy 
from torch_geometric.utils import is_undirected, to_undirected

def nearest_nodes_in_graph(G, X, limited_type = None):
    nodes = list(G.node_indices())
    if limited_type is not None:
        nodes = [n for n in G.node_indices() if G[n]['node_label'] == limited_type]
    G_feats = np.array(list(map(lambda a : (G[a]['x'], G[a]['y']), nodes)))
    tree = KDTree(G_feats)
    dist, ind = tree.query(X, return_distance=True, k=1)
    nodes = np.array(nodes)[ind.flatten()]
    return nodes, dist


def add_graph_components(df, G, node_label, limited_type = None):
    '''build bi-directed edges between nodes and road graphs '''
    pois_feats = np.atleast_2d(np.array(list(zip(df['x'], df['y']))))
    nearest_nodes, nearest_dist =  nearest_nodes_in_graph(G, pois_feats, limited_type = limited_type)
    node_info_ls = []
    for i in range(len(df)):
        node_info = dict(df.iloc[i, :])
        node_info['street_count'] = 1
        node_info['node_label'] = node_label
        node_info_ls.append(node_info)
    node_a_ls = G.add_nodes_from(node_info_ls)
    edges_info_ls = []
    for i in range(len(df)):
        node_a = node_a_ls[i]
        # add undirected edges
        node_b = nearest_nodes[i]
        dist = nearest_dist[i]
        edge_feats = {
                    'oneway': False,
                    'reversed': False,
                    'length': dist.item(),
                    'geometry': LineString([[node_info['x'], node_info['y']],  [G[node_b]['x'], G[node_b]['y']]]),
                    'travel_time': 0,  # not counting
                    'edge_label' : (node_label, G[node_b]['node_label']) }
        edges_info_ls.append((node_a, node_b, edge_feats))    
    G.add_edges_from(edges_info_ls)
    return G 

def steiner_T(G, terminal_type ='target',weight_l = 'length'):
    """simplified the graph with give terminals"""
    terminal_nodes = [i for i in G.node_indices() if  G[i]['node_label'] == terminal_type]
    ## add weight 
    def weight_fn(edge):
        return edge[weight_l]
    steiner_G = steiner_tree(G, terminal_nodes =terminal_nodes, weight_fn = weight_fn)
    return steiner_G


def query_road_graph(lat, lon, dist, to_csr, dist_type='bbox',
                                network_type='all', simplify = True, project = True, 
                                add_speed = True, tolerance =25, convert_to_rx = True, undirected = True):
    # get graph first 
    G= ox.graph.graph_from_point((lat, lon), 
                                dist=dist, dist_type=dist_type,
                                network_type=network_type, simplify=simplify)
    if project:
        G = ox.projection.project_graph(G, to_crs = to_csr, to_latlong=False)
    if simplify:
        G = ox.simplification.consolidate_intersections(G, tolerance=tolerance, 
                                                        rebuild_graph=True, 
                                                        dead_ends=False, 
                                                        reconnect_edges=True)
    if add_speed:
        G = ox.routing.add_edge_speeds(G)
        G = ox.routing.add_edge_travel_times(G)
    for i in G.nodes:
        G.nodes[i]['node_label'] = 'road'
        G.nodes[i]['split_type'] = -1 
    for e in G.edges:
        G.edges[e]['edge_label'] = ('road', 'road')
    if convert_to_rx:
        G = rx.networkx_converter(G, keep_attributes=True)
    if undirected:
        G = G.to_undirected(multigraph=False) 
    return G 

def mapping_edges(edges, if_homo = True):
    if not if_homo: # hetero
        nodes0, _= edges[0].unique().sort()
        nodes1, _= edges[1].unique().sort()
        e0 = map_index(edges[0],
                    nodes0,
                    max_index=max(nodes0),
                    inclusive=True)[0]
        e1 = map_index(edges[1],
                    nodes1,
                    max_index=max(nodes1),
                    inclusive=True)[0]
        return torch.stack([e0, e1], dim=0)
    if if_homo:
        nodes = edges.flatten().unique()
        e = map_index(edges.flatten(),
                    nodes,
                    max_index=max(nodes),
                    inclusive=True)[0].reshape(2, -1)
        return e

def pad( x, pad_shape = 64):
    if  x.shape[0]< pad_shape:
        x = torch.concat((x, torch.zeros(pad_shape-x.shape[0], x.shape[1])), dim = 0)
    return x     

def to_pyg(G, G_type, node_group_attrs, edge_group_attr, reshape_dict, scaler, scaler_label):
    if G_type == 'hetero':
        node_groups = {k : {'x' : []} for k in ['road', 'pois', 'target']}
        edge_groups = {('pois' , 'to', 'road') : {'edge_index' : [], 
                                                  'edge_attr' : []}, 
                       ('target' , 'to', 'road') : {'edge_index' : [], 
                                                  'edge_attr' : []}, 
                        ('road' , 'to', 'road') : {'edge_index' : [], 
                                                  'edge_attr' : []}
                       }
        data = copy.deepcopy(node_groups)
        data.update(edge_groups)
        nodes = list(G.node_indices())
        nodes.sort() # make sure all nodes mapping correct
        for n in nodes:
            l = G[n]['node_label']
            n_attrs = node_group_attrs[l]
            node_groups[l]['x'].append([float(G[n][k]) for k in n_attrs]) 
        for e_ix in G.edge_indices():
            e = G.edge_list()[e_ix]
            l0, l1 =  G[e[0]]['node_label'], G[e[1]]['node_label']
            assert l1 =='road' 
            edge_groups[(l0 , 'to', l1)]['edge_index'].append(e)
            edge_groups[(l0 , 'to', l1)]['edge_attr'].append([ float(G.get_edge_data_by_index(e_ix)[edge_group_attr])])
        # conver to torch
        for k, v in node_groups.items():
            col = reshape_dict[k]
            arr = np.zeros((col, ))
            if len(v['x']) >0:
                arr = np.array(v['x'])
                arr[:, :2] = scaler.transform(arr[:, :2], 'road_coords')
            data[k]['x'] = torch.tensor(arr).reshape(-1, col)
        for k, v in edge_groups.items():
            edges = torch.tensor(v['edge_index']).T.long().contiguous()
            edge_attr = torch.tensor([]).reshape(-1)
            if edges.shape[0] > 0:
                edges = mapping_edges(edges, if_homo = (k[0] == k[-1]))
                edge_attr = torch.tensor(scaler.transform(np.array(v['edge_attr']).reshape(-1,1), scaler_label)).reshape(-1).to(torch.float32)
                if k[0] == k[-1] and not is_undirected(edges, edge_attr):
                    edges, edge_attr = to_undirected(edges, edge_attr)
            v['edge_index'] = edges.reshape(2, -1)
            v['edge_attr'] = edge_attr
            data[k] = v 
        data = HeteroData(data)
        if G.attrs is not None:
            data.graph_type = int(float(G.attrs['split_type']))
            data.y = float(G.attrs['y'])
            data.radius = float(G.attrs['radius'])
            t_f = np.array([ float(eval(G.attrs['target_feats'])[k]) for k in node_group_attrs['target'] if k != 'train_price' ]).reshape(1, -1)
            t_f[:, :2] = scaler.transform(t_f[:, :2], 'road_coords')
            data.target_feats = torch.tensor(t_f).to(torch.float32)
        # pad the shape 
        pad_shape = data['target'].x.shape[0] 
        data['pois'].x = pad(data['pois'].x, pad_shape=pad_shape).to(torch.float32)
        data['road'].x = pad(data['road'].x, pad_shape=pad_shape).to(torch.float32)
        data['target'].x = data['target'].x.to(torch.float32)
        return data 
    
    elif G_type =='homo':
        data = Data()
        x = []
        y = []
        edge_index = []
        edge_attr = []
        x_type = []
        node_info_list = G.nodes()
        for n_info in node_info_list:
            feats = [n_info[k] for k in node_group_attrs] + [n_info['split_type']] 
            x.append(feats) 
            y.append(n_info['regression_target'])
            x_type.append(n_info['split_type'])
        edge_index = list(G.edge_list())
        edge_attr = [i[edge_group_attr] for i in G.edges()]
        edge_index = torch.tensor(edge_index).T.long().contiguous()
        data['x'] = torch.tensor(np.array(x)).to(torch.float32)
        data['y'] = torch.tensor(y).to(torch.float32).reshape(-1,1)
        data['edge_index'] = edge_index
        data['edge_attr'] = torch.tensor(np.array(edge_attr).reshape(-1,1)).to(torch.float32)
        # split train, val, test 
        all_nodes = np.array(list(range(len(x_type))))
        x_type =np.array(x_type)
        train_nodes = all_nodes[x_type ==0]
        val_nodes = all_nodes[x_type == 1]
        test_nodes = all_nodes[x_type == 2]
        tr_g = data.subgraph(subset = torch.tensor(train_nodes))
        val_g = data.subgraph(subset =  torch.tensor(np.concatenate([train_nodes, val_nodes])))
        test_g = data.subgraph(subset =  torch.tensor(np.concatenate([train_nodes, test_nodes])))
        tr_g.mask = tr_g.x[:, -1] == 0 
        tr_g.x = tr_g.x[:, :-1] # remove the split type 
        val_g.mask = val_g.x[:, -1] == 1 
        val_g.x = val_g.x[:, :-1] # remove the split type 
        test_g.mask = test_g.x[:, -1] == 2
        test_g.x = test_g.x[:, :-1] # remove the split type 
        # scale coords and weights 
        train_coords = tr_g.x[:, :2].numpy()
        scaler.update_scaler({ 'homo_coords' : train_coords})
        train_weights = tr_g.edge_attr.numpy().reshape(-1, 1)
        scaler.update_scaler({ 'weight' : train_weights})

        tr_g['x'] = torch.concat((torch.tensor(scaler.transform(tr_g['x'].numpy()[:, :2], 'homo_coords')).to(torch.float32), tr_g['x'][:, 2:]), dim=1)
        val_g['x'] = torch.concat((torch.tensor(scaler.transform(val_g['x'].numpy()[:, :2], 'homo_coords')).to(torch.float32), val_g['x'][:, 2:]), dim=1)
        test_g['x'] = torch.concat((torch.tensor(scaler.transform(test_g['x'].numpy()[:, :2], 'homo_coords')).to(torch.float32), test_g['x'][:, 2:]), dim=1)

        tr_g['edge_attr'] = torch.tensor(scaler.transform(tr_g['edge_attr'].numpy().reshape(-1,1), 'weight')).to(torch.float32)
        val_g['edge_attr'] = torch.tensor(scaler.transform(val_g['edge_attr'].numpy().reshape(-1,1), 'weight')).to(torch.float32)
        test_g['edge_attr'] = torch.tensor(scaler.transform(test_g['edge_attr'].numpy().reshape(-1,1), 'weight')).to(torch.float32)
        return tr_g, val_g, test_g
        