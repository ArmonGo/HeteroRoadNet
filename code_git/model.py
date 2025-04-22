from pykrige.rk import Krige
import numpy as np
from mgwr.gwr import GWR as Mod_GWR
from mgwr.sel_bw import Sel_BW
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import  check_is_fitted
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import   GATConv, HGTConv, HeteroConv, Linear, GCNConv

    
class OrdinaryKriging(BaseEstimator, RegressorMixin):
    def __init__(self, nlags=5, variogram_model='gaussian'):
        self.nlags = nlags
        self.variogram_model = variogram_model

    def split_coords(self, X, cols = [0,1]): # default of coordinates
        coords = X.iloc[:, cols]
        X = X.iloc[:, max(cols): ]
        return np.array(X), np.array(coords)
    
    def fit(self, X, y):
        X, coords = self.split_coords(X)
        self.model_ = Krige(nlags=self.nlags, variogram_model = self.variogram_model)
        self.model_.fit(np.array(coords),np.array(y))
        return self
    
    def predict(self, X):
        X, coords = self.split_coords(X)
        predictions = self.model_.predict(np.array(coords))
        return predictions


class GWR(BaseEstimator, RegressorMixin):
    def __init__(self, constant=True, kernel='gaussian', bw=None):
        self.constant = constant
        self.kernel = kernel
        self.bw = bw
        
    def split_coords(self, X, cols = [0,1]): # default of coordinates
        coords = np.array(X.iloc[:, cols])
        X = np.array(X.iloc[:, max(cols): ])
        rng = np.random.RandomState(42)
        rand = rng.randn(X.shape[0], X.shape[1])/10000 # to prevent a singular matrix
        return X+rand, coords
    
    def fit(self, X, y):
        X, coords = self.split_coords(X)
        if self.bw is None:
            self.bw = Sel_BW(coords, y.reshape((-1, 1)), X).search()
        self.model_ = Mod_GWR(coords, y.reshape((-1, 1)), X, self.bw, constant=self.constant, kernel=self.kernel)
        gwr_results = self.model_.fit()
        self.scale = gwr_results.scale
        self.residuals = gwr_results.resid_response 
        return self

    def predict(self, X):
        check_is_fitted(self)
        X, coords = self.split_coords(X)
        pred = self.model_.predict(coords, X, exog_scale=self.scale, exog_resid=self.residuals
               ).predictions
        return pred


#==============================================================================
# for comparison graphs 
#==============================================================================

class HomoGNN(torch.nn.Module):
    def __init__(self, out_channels_layer1, dropout_layer1, out_channels_layer2, dropout_layer2, lin_channels):
        super().__init__()
        self.output_lin = Linear(-1, 1)
        self.inter_lin = Linear(-1, lin_channels)
        self.relu = nn.ReLU()
        self.conv1 = GCNConv(in_channels=-1, out_channels=out_channels_layer1) 
        self.conv2 = GCNConv(in_channels=out_channels_layer1, out_channels=out_channels_layer2)
        self.dropout1 = nn.Dropout(p=dropout_layer1)
        self.dropout2 = nn.Dropout(p=dropout_layer2)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.inter_lin(x)
        x = self.relu(x)
        x = self.output_lin(x)
        return x  

#==============================================================================
# for road hetero graphs 
#==============================================================================

class HeteroGNN(torch.nn.Module):
    def __init__(self, node_types, edge_types, 
                 dict_out_channels,
                 pois_out_channels,  pois_drop_out, 
                 target_out_channels, target_drop_out,
                 road_out_channels, road_drop_out,
                 all_out_channels, all_drop_out
                 ):
        super().__init__()
        self.pois_convs =  nn.Sequential(
                                Linear(-1, pois_out_channels),
                                nn.ReLU(),
                                nn.Dropout(p=pois_drop_out), 
                                Linear(pois_out_channels, pois_out_channels // 2),
                                nn.ReLU()
                            ) 
        self.target_convs =  nn.Sequential(
                                Linear(-1, target_out_channels),
                                nn.ReLU(),
                                nn.Dropout(p=target_drop_out), 
                                Linear(target_out_channels, target_out_channels // 2),
                                nn.ReLU()
                            )
                      
        # define linear layers for input dictionary
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types: # node type
            self.lin_dict[node_type] = Linear(-1, dict_out_channels)
        
        # road weights layer 
        layer_info = {}
        for r in edge_types:
            if r == ('road', 'to', 'road'):
                layer_info[r] = GCNConv(in_channels=-1, out_channels= road_out_channels) 
            elif r == ('pois', 'to', 'road'):
                layer_info[r] =  GATConv(in_channels=-1, out_channels= road_out_channels, add_self_loops=False, heads = 1)
            elif r == ('target', 'to', 'road'):
                layer_info[r] =  GATConv(in_channels=-1, out_channels= road_out_channels, add_self_loops=False, heads = 1)
        self.conv_hetero = HeteroConv(layer_info)
        # road non weights layer
        self.conv_hgt = HGTConv(in_channels = -1, out_channels=  road_out_channels, metadata = (node_types, edge_types), heads = 1)
        self.road_dropout = nn.Dropout(p=road_drop_out)
        
        # output layer
        self.output_convs =  nn.Sequential(
                                Linear(-1, all_out_channels),
                                nn.ReLU(),
                                nn.Dropout(p=all_drop_out), 
                                Linear(all_out_channels, all_out_channels // 2),
                                nn.ReLU(),
                                nn.Dropout(p=all_drop_out),
                                Linear( all_out_channels // 2, 1),
                            )
                       
        
    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict, batch = data.x_dict, data.edge_index_dict, data.edge_attr_dict, data.batch_dict
        x_dict_all ={}
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }
        # road embeddings
        x_dict_weights = self.conv_hetero(x_dict, edge_index_dict, edge_attr_dict)
        x_dict_weights = {key: F.leaky_relu(v) for key, v in x_dict_weights.items()}
        x_dict_non_weights = self.conv_hgt(x_dict, edge_index_dict)
        x_dict_non_weights = {key: F.leaky_relu(v) for key, v in x_dict_non_weights.items()}
        x_dict_all['road']  = x_dict_weights['road'] + x_dict_non_weights['road']
        # target embeddings 
        x_dict_all['target']  = self.target_convs(x_dict['target'])
        # poi embedddings 
        x_dict_all['pois']  = self.pois_convs(x_dict['pois'])
        batch_x_dict = {}
        for node_type, x in x_dict_all.items():
            batch_x_dict[node_type]  = torch.cat([gmp(x, batch[node_type]), gap(x, batch[node_type])], dim=1)
        x = torch.concat(list(batch_x_dict.values()), dim = 1)
        x = self.output_convs(torch.concat([x, data.target_feats], 1))
        return x
