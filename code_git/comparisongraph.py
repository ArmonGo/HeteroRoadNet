import numpy as np
from sklearn.neighbors import KDTree
from graphoperation import add_graph_components, query_road_graph, steiner_T, to_pyg
from pointoperation import *
from utilis import * 
import torch 
import rustworkx as rx
import copy 
import json
import gzip
import os 


class ComparisonG:
    def __init__(self, tag, dataload_f, tolerance= 30, adaptive_radius = True, steiner_tree = True, save_p = None) -> None:
        # load target data 
        self.target_df, ( lat, lon, query_r, to_csr) = dataload_f(split_rate= (0.7, 0.1, 0.2), scale = True, add_train_price = True)
        self.target_df['nid'] = np.array(list(range(len(self.target_df))))
        # adaptive radius
        coords_train, y_train = np.array(self.target_df.loc[self.target_df['split_type'] == 0, ['x', 'y']]),  \
                                np.array(self.target_df.loc[self.target_df['split_type'] == 0, ['regression_target']])
        self.adp = None
        if adaptive_radius:
            self.adp = AdaptiveRadius(coords_train, y_train, min_samples_leaf = 30)
        # pois
        self.pois_df =  get_pois_df(tag, (lat,  lon), dist=query_r, to_csr = to_csr)
        self.pois_tree = PoisTree(self.pois_df)
        self.target_pois_df  = pd.DataFrame()
        for p_t in list(zip(self.target_df.x, self.target_df.y)):
            p = np.array(p_t).reshape(1, -1)
            dist = self.adp.get_radius(p)
            pois_feats = self.pois_tree.query_radius(p, r=dist, output = 'agg')
            self.target_pois_df = pd.concat([self.target_pois_df, pois_feats], axis=0)
        self.target_pois_df = pd.concat([self.target_df, self.target_pois_df.reset_index(drop=True)], axis=1)
        ## scale features 
        poi_col = list(pois_feats.columns)
        s = MinMaxScaler()
        s.fit(self.target_pois_df.loc[self.target_pois_df['split_type'] == 0, poi_col])
        self.target_pois_df.loc[:, poi_col] = self.target_pois_df.loc[:, poi_col].astype(float)
        self.target_pois_df.loc[:, poi_col] = s.transform(self.target_pois_df.loc[:, poi_col])
    
        # base graph 
        self.roadG = query_road_graph(lat, lon, query_r, to_csr, dist_type='bbox',
                    network_type='all', simplify = True, project = True, undirected =True,
                    add_speed=True, tolerance = tolerance, convert_to_rx = True)
        print('road collection done...')

        # base scaler
        road_coords = np.array([[self.roadG[i]['x'], self.roadG[i]['y']] for i in self.roadG.node_indices()])
        road_travel_length = np.array([[self.roadG.get_edge_data_by_index(e)['length']] for e in self.roadG.edge_indices()])
        self.scalers = ScaleTransformer( fitting_dict = {'road_coords': road_coords,
                                                         'length': road_travel_length
                                                         }, scaler = MinMaxScaler)

        # add targets into graph for subgnn
        self.roadG_sub = copy.deepcopy(self.roadG)
        self.roadG_sub = add_graph_components(self.target_df[self.target_df['split_type'] == 0], self.roadG_sub, 'target') # add the target in 
        print('add targets into graph for hetero gnn...')
        # add targets into graph for distance
        self.roadG_dis = copy.deepcopy(self.roadG)
        self.roadG_dis = add_graph_components(self.target_df, self.roadG_dis, 'target') # add the target in 
        print('add targets into graph for homo gnn...')
        if steiner_tree: # use steiner tree all the time to improve the speed
           self.roadG_sub = steiner_T(self.roadG_sub, terminal_type ='target')    
           self.roadG_dis = steiner_T(self.roadG_dis, terminal_type ='target')    
        # add poi 
        self.roadG_sub = add_graph_components(self.pois_df, self.roadG_sub, 'pois',  limited_type = 'road') # add the pois
        # distance 
        self.dis = DistanceMatrix(self.roadG_dis.copy(), filter_conds = {'node_label' : 'target'}) 
        # other settings
        self.save_p = save_p
        print('prepare done...')

    def dump_road_subg_json(self):  
        nodes = np.array(list(self.roadG_sub.node_indices()))
        target_nodes_ix = [j for j in nodes if self.roadG_sub[j]['node_label'] == 'target']
        road_tree = KDTree(np.array([[self.roadG_sub[i]['x'], self.roadG_sub[i]['y']] for i in self.roadG_sub.node_indices()]))
        # define functions for dumping edge, graph and node attrs
        def nodes_attr(node):
            out =  {k : str(node[k]) for k in node.keys()} # keep it all
            return out 
        def edges_attrs(edge):
            remain_feats = ['length', 'travel_time', 'edge_label']
            out = {k : str(edge[k]) for k in remain_feats}
            return out 
        def graphs_attr(g_attr):
            return {k: str(g_attr[k]) for k in g_attr.keys()}
        
        fl = self.save_p + '/graph.json.gz'
        # remove the file if exists
        try:
            os.remove(fl)
        except OSError:
            pass
        with gzip.open(fl, 'wt', encoding="utf-8") as file:
            for n in target_nodes_ix: # write training nodes 
                p = np.array([self.roadG_sub[n]['x'], self.roadG_sub[n]['y']]).reshape(1, -1)
                if self.adp is not None:
                    dist = self.adp.get_radius(p)
                else:
                    dist = 3000 
                retained_nodes = nodes[road_tree.query_radius(p, r=dist)[0]]
                retained_nodes= retained_nodes[retained_nodes != n] # remove centroid node  
                subg = copy.deepcopy(self.roadG_sub.subgraph(retained_nodes, preserve_attrs= True))
                # add graph attr
                subg.attrs = {}
                subg.attrs['y'] = self.roadG_sub[n]['regression_target']
                subg.attrs['split_type'] = self.roadG_sub[n]['split_type']
                subg.attrs['radius'] = dist 
                subg.attrs['target_feats'] = self.roadG_sub[n]
                # dump graph to json 
                j = rx.node_link_json(subg, node_attrs=nodes_attr, edge_attrs=edges_attrs, graph_attrs=graphs_attr)
                json.dump(j, file)
                file.write('\n')
            # write test and validation nodes 
            for n in range(self.target_df.shape[0]):
                if self.target_df.iloc[n, : ]['split_type'] == 0:
                    pass
                else:
                    p = np.array([self.target_df.iloc[n, : ]['x'], self.target_df.iloc[n, :]['y']]).reshape(1, -1)
                    if self.adp is not None:
                        dist = self.adp.get_radius(p)
                    else:
                        dist = 3000 
                    retained_nodes = nodes[road_tree.query_radius(p, r=dist)[0]]
                    subg = copy.deepcopy(self.roadG_sub.subgraph(retained_nodes, preserve_attrs= True))
                    # add graph attr
                    subg.attrs = {}
                    subg.attrs['y'] = self.target_df.iloc[n, :]['regression_target']
                    subg.attrs['split_type'] = self.target_df.iloc[n, :]['split_type']
                    subg.attrs['radius'] = dist 
                    subg.attrs['target_feats'] = dict(self.target_df.iloc[n, :])
                    # dump graph to json 
                    j = rx.node_link_json(subg, node_attrs=nodes_attr, edge_attrs=edges_attrs, graph_attrs=graphs_attr)
                    json.dump(j, file)
                    file.write('\n')
        return 'subgraphs dumped'
        
    
    def dump_road_subg_dataset(self, exclude_node_group_attrs, edge_group_attrs):
        dataset = {'train_subgs' : [], 
                       'val_subgs' : [], 
                       'test_subgs' : []
                       }
            # load node attrs and reshape dimensions
        node_group_attrs = {}
        with gzip.open( self.save_p + '/graph.json.gz', 'rt', encoding="utf-8") as file:
            for line in file: 
                json_obj = json.loads(line)
                subg = rx.parse_node_link_json(json_obj)
                for n in subg.nodes():
                    if n['node_label'] not in node_group_attrs.keys():
                        l = list(n.keys())
                        l = list(set(l).difference(exclude_node_group_attrs)) 
                        if n['node_label'] != 'target':
                            node_group_attrs[n['node_label']] = ['x', 'y'] + l
                        else: 
                            node_group_attrs[n['node_label']] = ['x', 'y', 'train_price'] + l # make sure xy train price always at the front 
                break 
        file.close() 
        reshape_dict = {'road' : len(node_group_attrs['road']), 
                        'pois' : len(node_group_attrs['pois']), 
                        'target' : len(node_group_attrs['target'])}  
    
        for e in edge_group_attrs:
            #  train, val, test 
            with gzip.open( self.save_p + '/graph.json.gz', 'rt', encoding="utf-8") as file:
                for line in file:
                    json_obj = json.loads(line)
                    subg = rx.parse_node_link_json(json_obj)
                    data = to_pyg(subg, 'hetero', node_group_attrs= node_group_attrs,
                                edge_group_attr = e, reshape_dict = reshape_dict, scaler=self.scalers, scaler_label= e)
                    if data.graph_type == 0:
                        dataset['train_subgs'].append(data)
                    elif data.graph_type == 1:
                        dataset['val_subgs'].append(data)
                    elif data.graph_type == 2:
                        dataset['test_subgs'].append(data)
                    else:
                        print('There is unknown graph type: ', data.graph_type)
                        raise KeyError
            torch.save(dataset, self.save_p + '/hetero_travel_'+ e + '.pt')


    def load_tabular_data(self):
        s = MinMaxScaler()
        feats = list(self.target_pois_df.columns)
        for i in ['split_type', 'regression_target', 'nid', 'train_price']:
            feats.remove(i)
        X_train = self.target_pois_df.loc[self.target_pois_df['split_type'] == 0, feats]
        X_val = self.target_pois_df.loc[self.target_pois_df['split_type'] == 1, feats]
        X_test = self.target_pois_df.loc[self.target_pois_df['split_type'] == 2, feats]
        y_train = self.target_pois_df.loc[self.target_pois_df['split_type'] == 0, ['regression_target']]
        y_val = self.target_pois_df.loc[self.target_pois_df['split_type'] == 1,  ['regression_target']]
        y_test = self.target_pois_df.loc[self.target_pois_df['split_type'] == 2,  ['regression_target']]
        X_train.loc[:, ['x', 'y']] = s.fit_transform(X_train.loc[:, ['x', 'y']])
        X_val.loc[:, ['x', 'y']] = s.transform(X_val.loc[:, ['x', 'y']])
        X_test.loc[:, ['x', 'y']] = s.transform(X_test.loc[:, ['x', 'y']])
        return X_train, X_val, X_test, y_train, y_val, y_test

    def homograph(self, threshold = 0.1, use_pois=False, use_road = False, node_group_attrs =None, edge_group_attr = None):
        if not use_road:
            d = 'euclidean'
        else:
            d = 'length'
        s_nodes, pairs, weights = self.dis.calculate_distance(threshold, distance = d)
        if not use_pois:
            feats_df = self.target_df
        else:
            feats_df = self.target_pois_df
        mapping_nodes = {} 
        g = rx.PyGraph()
        for n in s_nodes:
            node_info = dict(feats_df.iloc[int(self.roadG_dis[n]['nid']), : ])
            new_id = g.add_node(node_info)
            mapping_nodes[n] = new_id
        for ix in range(len(pairs)):
            [e_0, e_1] = pairs[ix]
            e_0 = mapping_nodes[e_0]
            e_1 = mapping_nodes[e_1]
            w = weights[ix]
            g.add_edge(e_0, e_1, { 'weight' : w})
        # convert pyg graph 
    
        tr_g, val_g, test_g = to_pyg(g, 'homo', 
                                     node_group_attrs, edge_group_attr, 
                                     reshape_dict =None, scaler = self.scalers, scaler_label = None)
        return tr_g, val_g, test_g 


    def main(self,  graph_const = ['hetero', 'homo', 'tabular'],
                    exclude_node_group_attrs =  ['split_type', 'regression_target', 'nid', 'osmid_original', 
                                                'street_count', 'node_label', '__networkx_node__', 'geometry', 'osmid', 
                                                'lon', 'lat', 'highway', 'x', 'y', 'ref', 'train_price'], 
                    edge_group_attrs = ['length'], homo_threshold = 0.001):

        if 'tabular' in graph_const:
            dataset = self.load_tabular_data()
            torch.save(dataset, self.save_p + '/tabular_data.pt')

        if 'hetero' in graph_const:
            self.dump_road_subg_json()
            self.dump_road_subg_dataset(exclude_node_group_attrs, edge_group_attrs)
            
        if 'homo' in graph_const:
            # no poi, no roads
            l  = list(self.target_df.keys())
            l = list(set(l).difference(exclude_node_group_attrs)) 
            l = ['x', 'y'] + l
            tr_g, val_g, test_g  = self.homograph( threshold = homo_threshold, use_pois=False, use_road = False, node_group_attrs =l, edge_group_attr = 'weight')
            torch.save((tr_g, val_g, test_g), self.save_p + '/homo_euclidean_data.pt')
            # no poi, roads
            tr_g, val_g, test_g  = self.homograph(threshold = homo_threshold, use_pois=False, use_road = True, node_group_attrs =l, edge_group_attr = 'weight')
            torch.save((tr_g, val_g, test_g), self.save_p + '/homo_length_data.pt')
            # with poi, no roads
            l  = list(self.target_pois_df.keys())
            l = list(set(l).difference(exclude_node_group_attrs)) 
            l = ['x', 'y'] + l
            tr_g, val_g, test_g  = self.homograph(threshold = homo_threshold, use_pois=True, use_road = False, node_group_attrs =l, edge_group_attr = 'weight')
            torch.save((tr_g, val_g, test_g), self.save_p + '/homo_euclidean_poi_data.pt')
            # with poi, with roads
            tr_g, val_g, test_g  = self.homograph(threshold = homo_threshold, use_pois=True, use_road = True, node_group_attrs =l, edge_group_attr = 'weight')
            torch.save((tr_g, val_g, test_g), self.save_p + '/homo_length_poi_data.pt')
        return 'data done!'
    

