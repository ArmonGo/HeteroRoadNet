
from experiement import Experiment
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import  mean_absolute_percentage_error
from model import GWR,  OrdinaryKriging
from utilis import mape
import numpy as np 

dataload_p = './YOUR_LOCATION/'
result_save_p = './YOUR_LOCATION/'
train_params = {
                'epochs' : 3000,
                'device' : 'cuda',
                'tuning_epochs' : 100,
                'tuning_trails' : 100, 
                'batch_size' : 128, 
                'loss_f': mape,
                'ml_score_f' : mean_absolute_percentage_error}

comparisonG_params =  ['homo_euclidean_data', 'homo_length_data', 'homo_euclidean_poi_data', 'homo_length_poi_data']
earlystop_params = {'patience' : 100,
                     'min_delta' : 0, 
                     'save_best_path' : './YOUR_LOCATION/'}

ml_params = {
                'Lr_ridge' : {'regressor' : Ridge, 
                        'searching_param' : {"alpha": np.arange(0.1,1,0.1)}
                        },
                'RandomForest' : {'regressor' : RandomForestRegressor, 
                        'searching_param' :  {
                                              "min_samples_split" : [2,3,5],
                                              "min_samples_leaf" : [3,5,10]
                                             }
                                 },     
                'XGBoost' : {'regressor' : XGBRegressor, 
                        'searching_param' :  { "learning_rate" : [0.1, 0.01, 0.005],
                                                "reg_alpha" : np.arange(0, 1.1, 0.1),
                                                "reg_lambda" : np.arange(0, 1.1, 0.1)
                                            } 
                                 },
                'LightGBM' : {'regressor' : LGBMRegressor, 
                                'searching_param' : {
                                                    "reg_alpha" : np.arange(0, 1.1, 0.1),
                                                    "reg_lambda" : np.arange(0, 1.1, 0.1),
                                                    "learning_rate" : [0.1, 0.01, 0.005],
                                                    "verbose": [-100]
                                                    }     
                             },
                'GWR' : {'regressor' : GWR, 
                         'searching_param' : {"constant": [True]}  
                         },

                "Kriging": { 'regressor' : OrdinaryKriging, 
                            "searching_param": {"nlags" : range(10,130,20),
                                               "variogram_model":[ "gaussian", "linear"] } #  "spherical", "power"
                                }
                }


if __name__ == "__main__":
    exp = Experiment( dataload_p, result_save_p, train_params, comparisonG_params, earlystop_params, ml_params)
    exp.main(repeat= 10)

