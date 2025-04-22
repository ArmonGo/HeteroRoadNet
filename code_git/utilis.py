from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np 
import osmnx as ox 
import pyproj
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
import rustworkx as rx 
import copy
from sklearn.model_selection import ParameterGrid
import lightgbm as lgb 
import torch 
import torch.nn.functional as F
import warnings

class AdaptiveRadius:
    def __init__(self, coords_train, y_train, min_samples_leaf = 30):
        self.clf = tree.DecisionTreeRegressor(min_samples_leaf = min_samples_leaf)
        self.clf.fit(coords_train, y_train)
        pred_y_train = self.clf.predict(coords_train)
        self.clusters = {k : [] for k in list(set(pred_y_train))}
        for i in range(len(pred_y_train)):
           self.clusters[pred_y_train[i]].append(coords_train[i])
        # calculate the radius
        self.radius_dict = {k : self.calculate_radius_pairwise_avg(self.clusters[k])[0] for k in self.clusters.keys()}
    
    def calculate_radius_pairwise_avg(self, points):
        centroid = np.mean(points, axis=0)
        distances = euclidean_distances(points)
        radius = np.average(distances[np.triu_indices(distances.shape[0], k = 1)])
        return radius, centroid
    
    def get_radius(self, point):
        pred = self.clf.predict(point)
        return self.radius_dict[pred.item()]

    def plot_points_and_circle(self, points):
        radius, centroid =  self.calculate_radius_pairwise_avg(points) 
        plt.figure(figsize=(8, 8))
        plt.scatter(points[:, 0], points[:, 1], color='blue', label='Nodes')
        plt.scatter(centroid[0], centroid[1], color='red', label='Centroid')
        # Create a circle with the calculated radius
        circle = plt.Circle(centroid, radius, color='green', fill=False, linestyle='--', linewidth=2, label=f'Radius = {radius:.2f}')
        plt.gca().add_patch(circle)
        # Set plot limits and labels
        plt.xlim(centroid[0] - radius - 1, centroid[0] + radius + 1)
        plt.ylim(centroid[1] - radius - 1, centroid[1] + radius + 1)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Point Cloud with Radius')
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()

class DistanceMatrix:
    def __init__(self, G, filter_conds = {'node_label' : 'target'} ):
        self.G = G
        k, v = list(filter_conds.keys())[0], list(filter_conds.values())[0]
        self.node_ix = []
        self.split_type = []
        self.feats = []
        for n in self.G.node_indices():
            if self.G[n][k] == v :
                self.node_ix.append(n)
                self.split_type.append(self.G[n]['split_type'])
                self.feats.append([self.G[n]['x'], self.G[n]['y']])
        self.node_ix = np.array(self.node_ix)
        self.split_type = np.array(self.split_type)
        self.feats = np.array(self.feats)
        
    def euclidean_distance(self):
        m = euclidean_distances(self.feats, self.feats)
        return m

    def graph_distance(self, w_l):
        '''calculate the road distance between node pairs. 
        The graph itself is undirected and full connected as 
        we remove all dead endpoints and simplified the graph '''
        m = []
        def w_fn(edge):
            return edge[w_l]
        for i in self.node_ix:
            d = rx.dijkstra_shortest_path_lengths(self.G, i, w_fn)
            row = [ d[j] if j != i else 0 for j in self.node_ix]
            m.append(row)
        m = np.array(m)
        return m
    
    def calculate_distance(self, percentile, distance = 'euclidean'):
        if distance == 'euclidean':
            m = self.euclidean_distance()
        elif distance == 'length':
            m = self.graph_distance(w_l= distance)
        else: 
            warnings.warn(f"{distance} choice does not exist")
            raise KeyError
        # filter matrix 
        s_nodes = self.node_ix
        t_nodes = self.node_ix[self.split_type == 0]
        m = m[:, self.split_type == 0] # convert to s-t distance matrix
        threshold = np.quantile(m, percentile, axis=1).reshape(-1, 1)
        m_ix = m - threshold
        pairs = []
        weights = []
        for i in range(len(s_nodes)):
            ix = np.where(m_ix[i] < 0)[0]
            pairs_l = t_nodes[ix]
            weights_l = m[i][ix]
            for p, w in zip(pairs_l, weights_l):
                pairs.append([s_nodes[i], p])
                weights.append(w)
                if s_nodes[i] not in t_nodes: # make sure undirected edges
                    pairs.append([p, s_nodes[i]])
                    weights.append(w)
        return s_nodes, pairs, weights
    
class ScaleTransformer:
    def __init__(self, fitting_dict = None, scaler = MinMaxScaler) -> None:
        # scalers dictionary 
        self.scalers  = {}
        if fitting_dict is not None:
            self.update_scaler(fitting_dict, scaler)

    def update_scaler(self, fitting_dict, scaler = MinMaxScaler):
        if scaler is None:
            scaler = MinMaxScaler # default 
        for k, v in fitting_dict.items():
            if k in ['length', 'travel_time', 'weight']: # make sure the weight is always bigger than 0
                s = scaler(feature_range=(0.001, 1.001), clip=True)
            else:
                s = scaler()
            s.fit(v)
            self.scalers[k] = s

    def transform(self, X, k):
        s = self.scalers[k]
        return s.transform(X)


def get_local_crs(lat, lon, radius):  
    trans = ox.utils_geo.bbox_from_point((lat, lon), dist = radius, project_utm = True, return_crs = True)
    to_csr = pyproj.CRS( trans[-1])
    return to_csr



class GridSearcher:
    def __init__(self, grid, score_f):
        self.param_grid =  ParameterGrid(grid)
        self.score_f = score_f
        self.best_score = np.inf
        self.best_param = None
        self.best_model = None 

    def search(self, rg, df_train, df_val, y_train, y_val):
        for param in self.param_grid:
            input_p = copy.deepcopy(param)
            regressor = rg() # instantiate the regressor 
            regressor.set_params(**input_p)
            regressor.fit(df_train, y_train)
            # count score
            pred = regressor.predict(df_val)
            s = self.score_f(y_val, pred)
            if s < self.best_score:
                self.best_score = s
                self.best_param = param
                self.best_model = copy.deepcopy(regressor)
        return self.best_score, self.best_param, self.best_model
    
    def search_tree_ensemble(self, rg, df_train, df_val, y_train, y_val, choice):
        for param in self.param_grid:
            regressor = rg() # instantiate the regressor 
            regressor.set_params(**param)
            if choice == 'LightGBM':
                callbacks = lgb.early_stopping(stopping_rounds = 100, first_metric_only = True, verbose = False, min_delta=0.0)
                regressor.fit(df_train, y_train, callbacks = [callbacks],  # use the best iteration to predict
                                eval_metric ='mape', eval_set =[(df_val, y_val)])
            elif choice == 'XGBoost':
                # callbacks = xgb.callback.EarlyStopping(rounds=100, metric_name='mape', data_name='validation_0', save_best=True)  
                # regressor.set_params(**{'callbacks' : [callbacks]})
                #regressor.fit(df_train, y_train,  # return the best model
                #             eval_set =[(df_val, y_val)])
                regressor.set_params(**{'eval_metric' : 'mape',
                                        'early_stopping_rounds' : 5})
                regressor.fit(df_train, y_train,  # return the best model
                              eval_set =[(df_val, y_val)], verbose= False)
            # count score
            pred = regressor.predict(df_val)
            s = self.score_f(y_val, pred)
            if s < self.best_score:
                self.best_score = s
                self.best_param = param
                self.best_model = copy.deepcopy(regressor)
        return self.best_score, self.best_param, self.best_model



#=====================================
# model training, validate, and predict
#=====================================

def mape(y_pred, y_true):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    return torch.mean((y_true - y_pred).abs() / y_true.abs())

def rmse(y_pred, y_true):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    return torch.sqrt(F.mse_loss(y_pred, y_true))


def train(model, device, optimizer, dataloader = None, data = None, loss_f = rmse):
    model.train()
    optimizer.train()
    if dataloader is None: # no batches
        assert data is not None
        data.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = loss_f(y_pred, data.y)
        loss.backward()
        optimizer.step()
        return loss
    else: # use batch 
        num_graphs = 0
        loss_all = 0
        for data in dataloader:
            data.to(device)
            optimizer.zero_grad()
            y_pred = model(data)
            loss = loss_f(y_pred, data.y)
            loss.backward()
            loss_all += data.num_graphs * loss.detach()
            num_graphs += data.num_graphs
            optimizer.step()
        return loss_all / num_graphs # average loss 

def evaluate(model, device, optimizer, dataloader = None, data = None, loss_f = rmse):
    model.eval()
    optimizer.eval()
    if dataloader is None:
        assert data is not None # comparison graphs 
        data.to(device)
        y_pred = model(data)[data.mask]
        y_true = data.y[data.mask]
        loss  = loss_f(y_pred, y_true).detach()
        return loss
    else:
        out = []
        y = []
        for data in dataloader:
            data.to(device)
            pred = model(data)
            out = out + [i for i in pred.flatten()]
            y = y + [i for i in data.y.flatten()]
        y_pred = torch.tensor(out)
        y_true = torch.tensor(y)
        loss = loss_f(y_pred, y_true).detach()
        return loss, y_pred, y_true
