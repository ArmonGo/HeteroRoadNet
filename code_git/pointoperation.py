
import osmnx as ox 
import numpy as np
import pandas as pd 
from shapely import Point
from sklearn.neighbors import KDTree
import math

def project_nodes_csr(df, to_csr):
    """project geometry points into cartasian coordinates"""
    df['geometry'] = list(map(lambda x, y: Point(x, y), df['lon'],df['lat'] ))
    df['geometry'] = list(map(lambda x : ox.projection.project_geometry(x, to_crs=to_csr)[0], df['geometry'] ))
    df['y'], df['x'] = np.array(list(map(lambda x : x.y, df.geometry))), \
                                            np.array(list(map(lambda x : x.x, df.geometry)))
    df = df.drop(columns = 'geometry')
    return df

def get_pois_df(tag, centroid, dist, to_csr):
    '''give centroid and query the nearby pois and convert into features with poi types indicator'''
    df = ox.features.features_from_point(centroid, tag, dist=dist)
    df = df.reset_index()
    keep_cols = [i for i in df.columns if i in [ 'element_type', 'osmid', 'geometry']
                                               or i in tag.keys() ]
    df = df[keep_cols]
    # convert polygon to points
    df.loc[ df['element_type'] != 'node','geometry'] = list(map( lambda x: x.centroid, df[df['element_type'] != 'node']['geometry']))
    df['geometry'] = list(map(lambda x : ox.projection.project_geometry(x, to_crs=to_csr)[0], df['geometry'] )) 
    df['y'], df['x'] = np.array(list(map(lambda x : x.y, df.geometry))), \
                                            np.array(list(map(lambda x : x.x, df.geometry)))
    
    # one hot encoding 
    encode_col = [i for i in df.columns if i not in ['osmid', 'geometry', 'y', 'x']]
    df = pd.get_dummies(df, columns = encode_col, dtype=int)
    df['split_type'] = -1 # not target and no issues of sliting train, val and test
    return df 

class PoisTree:
    def __init__(self, poi_df) -> None:
        poi_locs = np.array(poi_df[['x', 'y']])
        self.tree = KDTree(poi_locs)
        self.poi_df = poi_df
        self.feats = list(poi_df.columns)
    def query_radius(self, p, r, output = 'raw', outfeats = None):
        ind = self.tree.query_radius(p.reshape(1, -1), r)[0]
        if output == 'raw':
            outfeats = list(set(self.feats).difference(['osmid', 'geometry']))
            return self.poi_df.iloc[ind, : ][outfeats]
        elif output == 'agg': # pois sum
            outfeats = list(set(self.feats).difference(['osmid', 'geometry', 'lat', 'lon', 'y', 'x', 'split_type']))
            return pd.DataFrame([self.poi_df.loc[ind, : ][outfeats].sum(0)])


# Function to calculate the distance between two geographic coordinates using the Haversine formula
def haversine(coord1, coord2):
    R = 6371.0  # Radius of Earth in kilometers
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Function to find the centroid and radius
def find_centroid_and_radius(coords):
    # Calculate the centroid (average latitude and longitude)
    avg_lat = sum(coord[0] for coord in coords) / len(coords)
    avg_lon = sum(coord[1] for coord in coords) / len(coords)
    centroid = (avg_lat, avg_lon)
    # Calculate the radius (maximum distance from centroid to any point)
    radius = max(haversine(centroid, coord) for coord in coords) * 1000
    return centroid, np.ceil(radius) + 5000 # add buffer 

def is_within_radius(coord, centroid, radius):
    distance = haversine(centroid, coord)*1000
    return distance <= radius