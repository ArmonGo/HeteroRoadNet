import torch 
from model import * 
from torch_geometric.loader import DataLoader
from utilis import GridSearcher
import sys 
from paratune import objective_hetero, objective_homo
import optuna 
import re 
import schedulefree
from utilis import train, evaluate
import copy 

# ------------------------------
# early stop 
# ------------------------------

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, save_best_path = None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.save_best_path = save_best_path

    def early_stop(self, validation_loss, best_model = None):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            if best_model is not None:
                assert self.save_best_path is not None and self.save_best_path != ''
                torch.save(best_model, self.save_best_path + '/best_model_val.pt')
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

class Experiment:
    def __init__(self, dataload_p, result_save_p, train_params, comparisonG_params= None, earlystop_params =None, ml_params = None):
        self.dataload_p = dataload_p
        self.result_save_p = result_save_p
        self.comparisonG_params = comparisonG_params
        self.earlystop_params = earlystop_params
        self.ml_params = ml_params
        # train params
        self.epochs = train_params['epochs']
        self.device =  train_params['device']
        self.tuning_epochs = train_params['tuning_epochs']
        self.tuning_trails = train_params['tuning_trails']
        self.batch_size = train_params['batch_size']
        self.loss_f = train_params['loss_f']
        self.ml_score_f = train_params['ml_score_f']
        # result log
        self.rst = {}

    def run_homo(self, repeat = 1):
        assert self.comparisonG_params is not None
        for k in self.comparisonG_params:
            tr_g, val_g, test_g = torch.load(self.dataload_p + '/' + k +'.pt')
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective_homo(trial, tr_g, val_g, self.tuning_epochs, self.device, self.loss_f), 
                        n_trials=self.tuning_trails, timeout=None) 
            trial = study.best_trial
            print("  Value: ", trial.value)
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
            best_param = trial.params
            for t in range(repeat):
                print(k,': ', t)
                model = HomoGNN(
                              best_param['out_channels_layer1'],
                              best_param['dropout_layer1'],
                              best_param['out_channels_layer2'],
                            best_param['dropout_layer2'],
                            best_param['lin_channels'])
                model.to(self.device)
                # early stop 
                earlystopper = None
                if self.earlystop_params is not None:
                    earlystopper = EarlyStopper(**self.earlystop_params)
                # train             
                optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=best_param['lr'])
                for i in range(self.epochs):
                    train_loss = train(model, self.device, optimizer, dataloader = None, data = tr_g, loss_f = self.loss_f)
                    if i % 100 == 0:
                        print('train loss: ', train_loss)
                    if earlystopper is not None:
                        validation_loss = evaluate(model, self.device, optimizer, dataloader = None, data = val_g, loss_f = self.loss_f)
                        if_stop = earlystopper.early_stop( validation_loss, best_model =  model)
                        if if_stop:
                            print('early stop at epoches: ', i, ', stop val loss: ', validation_loss, ' min val loss: ', earlystopper.min_validation_loss)
                            break 
                # predict
                if self.earlystop_params is not None:
                    p = self.earlystop_params['save_best_path']
                    model = torch.load(p + '/best_model_val.pt')
                loss = evaluate(model, self.device, optimizer, dataloader = None, data = test_g, loss_f = self.loss_f) 
                if k not in self.rst.keys():
                    self.rst[k] = (loss, model, best_param)
                else:
                    if loss < self.rst[k][0]:
                        self.rst[k] = (loss, model, best_param)
            torch.save(self.rst, self.result_save_p + '/model_performance.pt')

    def run_hetero(self, weight_choice='travel_length', repeat = 1):
        dataset = torch.load(self.dataload_p + 'hetero_' + weight_choice + '.pt', weights_only = False)
        # define the dataloader
        train_loader = DataLoader(dataset['train_subgs'], batch_size=self.batch_size, shuffle= True)
        val_loader = DataLoader(dataset['val_subgs'], batch_size=self.batch_size, shuffle= True)
        test_loader = DataLoader(dataset['test_subgs'], batch_size=self.batch_size, shuffle= True)
        # param tuning 
        node_types = dataset['train_subgs'][0].node_types
        edge_types = dataset['train_subgs'][0].edge_types
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective_hetero(trial, train_loader, val_loader, node_types, edge_types, self.tuning_epochs, self.device, self.loss_f), 
                       n_trials=self.tuning_trails, timeout=None) 
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        best_param = trial.params
        for t in range(repeat):
            print('roadgnn',': ', t)
            model = HeteroGNN(node_types, edge_types, 
                              best_param['dict_out_channels'],
                              best_param['pois_out_channels'],
                              best_param['pois_drop_out'],
                            best_param['target_out_channels'],
                            best_param['target_drop_out'],
                            best_param['road_out_channels'],
                            best_param['road_drop_out'],
                            best_param['all_out_channels'],
                            best_param['all_drop_out'])
            model.to(self.device)
            # early stop 
            earlystopper = None
            if self.earlystop_params is not None:
                earlystopper = EarlyStopper(**self.earlystop_params)
            # train
            optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=best_param['lr'])   
            for i in range(self.epochs):
                train_loss = train(model, self.device, optimizer, dataloader = train_loader, data = None, loss_f = self.loss_f)
                if i % 100 == 0:
                    print(f"{train_loss =}")
                if earlystopper is not None:
                    val_loss, y_pred, y_true = evaluate(model, self.device, optimizer, dataloader = val_loader, data = None, loss_f=self.loss_f)
                    if_stop = earlystopper.early_stop( val_loss, best_model =  model)
                    if if_stop:
                        print('early stop at epoches: ', i, ', stop val loss: ', val_loss, ' min val loss: ', earlystopper.min_validation_loss)
                        break 
            
            # predict
            if self.earlystop_params is not None:
                p = self.earlystop_params['save_best_path']
                model = torch.load(p + '/best_model_val.pt')
                print(' best model has been loaded!')
            pred_loss, y_pred, y_true = evaluate(model, self.device, optimizer, dataloader = test_loader, data = None, loss_f=self.loss_f)
            if 'roadgnn_'+ weight_choice not in self.rst.keys():
                    self.rst['roadgnn_'+ weight_choice] = (y_pred, y_true, pred_loss, model, best_param)
            else:
                if pred_loss < self.rst['roadgnn_'+ weight_choice][2]:
                    self.rst['roadgnn_'+ weight_choice] = (y_pred, y_true, pred_loss, model, best_param)
            torch.save(self.rst, self.result_save_p + '/model_performance.pt')


    def  ml_run(self):
        # prepare data 
        X_train, X_val, X_test, y_train, y_val, y_test  = torch.load(self.dataload_p + '/'  +'tabular_data.pt')
        X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        X_val = X_val.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        X_test = X_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        if 'index' in X_train.columns:
            X_train = X_train.drop(columns = 'index')
            X_val = X_val.drop(columns = 'index')
            X_test = X_test.drop(columns = 'index')
        y_train, y_val, y_test = np.array(y_train).reshape(-1), np.array(y_val).reshape(-1), np.array(y_test).reshape(-1)

        for k, params in self.ml_params.items():
            print(k)
            regressor = params['regressor']
            searching_param = params['searching_param']
            try:
                searcher = GridSearcher(searching_param, self.ml_score_f)
            except:
                print("error:", sys.exc_info()[0])
            if k not in ['LightGBM', 'XGBoost']:
                best_score, best_param, best_model = searcher.search(regressor, X_train, X_val, y_train, y_val)
            else:
                best_score, best_param, best_model = searcher.search_tree_ensemble(regressor, X_train, X_val, y_train, y_val, choice = k)
            pred = best_model.predict(X_test)
            rst = self.ml_score_f(y_test, pred)
            self.rst[k] = (pred, y_test, rst, best_param)
            torch.save(self.rst, self.result_save_p + '/model_performance.pt')
        return 'ml done!'
   
    def main(self, repeat = 1):
        print('experiment begins!')
        self.run_homo( repeat= repeat)
        self.run_hetero(weight_choice = 'travel_length', repeat= repeat)
        self.ml_run()
        for k, v in self.rst.items():
            if 'homo' in k:
                print(k, v[0])
            else:
                print(k, v[2])
        