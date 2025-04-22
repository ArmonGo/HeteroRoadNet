# data reader
import pandas as pd
import numpy as np
import copy 
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from category_encoders.cat_boost import CatBoostEncoder
import kagglehub
from utilis import get_local_crs
from pointoperation import find_centroid_and_radius, project_nodes_csr
        
def load_data_path(path, target_tb, format = 'csv'):
    path = kagglehub.dataset_download(path)
    if format =='csv':
        df = pd.read_csv(path + '/' +target_tb,encoding_errors='ignore')
    else:
        arrays = dict(np.load(path + '/' +target_tb))
        data = {k: [s.decode("utf-8") for s in v.tobytes().split(b"\x00")] if v.dtype == np.uint8 else v for k, v in arrays.items()}
        df = pd.DataFrame.from_dict(data)
    return df 


def scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type', clip_col = 'regression_target', clip_interval = 0.01):
    cols = df.columns
    if scaler is not None:
        s = scaler()
    else: # default
        s = MinMaxScaler()
    if skip_coords:
        cols = [col for col in df.columns if col not in ['lon', 'lat', 'x', 'y',  mask_col]]
    if mask_col is not None:
        X = df[df[mask_col] == 0][cols]
    else:
        X = df[cols]
    s.fit(X)
    df.loc[:, cols] = df.loc[:, cols].astype(float)
    df.loc[:, cols] = s.transform(df.loc[:, cols] )
    if clip_col is not None: # target values has no zero for mape 
        df[clip_col] = df[clip_col] + clip_interval 
    return df 
        
    
def train_val_test_split(split_rate, length, shuffle = False, return_type = 'feats'):
    tr_r, val_r, te_r = split_rate
    assert tr_r + val_r + te_r == 1
    if shuffle:
        indices = np.random.permutation(length)
    else:
        indices = np.arange(length)
    ix_ls = [indices[:int(tr_r*length)], indices[int(tr_r*length):int((val_r + tr_r)*length)], indices[int((val_r + tr_r)*length):]]
    if return_type == 'index':
        mask_ls = []
        for i in range(3):
            mask = np.zeros(length, dtype=bool)
            mask[ix_ls[i]] = True
            mask_ls.append(mask)
        return mask_ls
    elif return_type == 'feats':
        split_type =  np.zeros(length, dtype=int)
        for i in range(3):
            split_type[ix_ls[i]] = i # 0-train, 1-val, 2-test
        return split_type

 
def load_melbourne_example( split_rate=None, scale =True, add_train_price = False):
    df_raw =  load_data_path("dansbecker/melbourne-housing-snapshot", 'melb_data.csv')
    df_raw = df_raw[df_raw['Landsize'] > 0]
    df = copy.deepcopy(df_raw[['Rooms', 'Type',  'Method',
       'Date', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt']])
    category_feats = ['Type','Method']

    df["regression_target"] = df_raw['Price']/df_raw['Landsize']
    df["lon"] = df_raw["Longtitude"] 
    df["lat"] = df_raw['Lattitude']
    df["Date"] = pd.to_numeric(pd.to_datetime(df_raw['Date'], dayfirst = True).dt.strftime('%Y-%m-%d').str.replace('-',''), errors='coerce')
    for i in df.columns:
        if i not in category_feats:
            df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df = df[(stats.zscore(df["regression_target"])<5) & (stats.zscore(df["regression_target"])>-2)]
    df = df.sort_values(by=['Date']) # temporal split 

    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
    coordinates = list(zip(df.lat , df.lon))
    (lat, lon), query_r = find_centroid_and_radius(coordinates)
    to_csr =  get_local_crs(lat,  lon, query_r) 
    df = project_nodes_csr(df, to_csr=to_csr)
    if len(category_feats)>0:
        encoder = CatBoostEncoder(cols = category_feats)
        encoder.fit(df[df['split_type'] == 0], df.loc[df['split_type'] == 0, 'regression_target'])
        df = encoder.transform(df)
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    if add_train_price:
        avg_price = np.mean(df[df['split_type'] == 0]['regression_target'])
        df['train_price'] = df['regression_target'] 
        df.loc[df['split_type'] != 0, 'train_price'] = avg_price
    df = df.drop(columns= ['lon', 'lat'])
    return df.reset_index(drop=True), ( lat, lon, query_r, to_csr)
