from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch
import pyvista as pv
from pathlib import Path
import csv
import re
import numpy as np
from collections import defaultdict

plys_to_poi_filename = { 'Bohunician': 'Bohunician', 'PunchedBlade': 'PunchedBlade', 'Shea_Levallois': 'SheaLevallois'}
base_path = Path('/mnt/materials/SIRF/MathPlusBerlin/DATA/LihicFlakesPOI/')

def label_to_data_object(dataset, label):
    filename = label_to_filename_map(dataset)[label][0]
    mesh = pv.read(base_path / f'3DModels_plys/{dataset}_plys/{filename}')
    pos = torch.tensor(mesh.points, dtype=torch.float32)
    face = torch.tensor(mesh.faces.reshape(-1,4)[:,1:].T, dtype=torch.long)
    norm = torch.tensor(mesh.point_normals, dtype=torch.float32)
    
    transform = T.Compose([T.FaceToEdge(remove_faces=False), T.PointPairFeatures()])
    graph = Data(pos=pos, face=face, norm=norm)
    
    return transform(graph)

def get_labels(dataset):
    poi = plys_to_poi_filename[dataset]
    POI_file = base_path / f'POIs_csv/{poi}_POIs_Point.csv'
    with open(POI_file) as f:
        sheet = list( csv.reader(f) )
    labels = [re.split(r'_|-|\s', ln[1])[0] for ln in sheet[1:]]
    
    return labels
    
def label_to_filename_map(dataset):
    ply_path = base_path / f'3DModels_plys/{dataset}_plys/'
    
    labels = get_labels(dataset)
    label_int = {l: int(l[1:]) for l in labels}
    label_int_to_filename = defaultdict(list)
    for p in ply_path.glob('*.ply'):
        if len(re.findall(r'\d+', p.name)) > 0:
            label_int_to_filename[int(re.findall(r'\d+', p.name)[0])].append(p.name)
            
    return {l: label_int_to_filename[label_int[l]] for l in labels}

def label_to_poi_map(dataset):
    poi = plys_to_poi_filename[dataset]
    POI_file = base_path / f'POIs_csv/{poi}_POIs_Point.csv'
    labels = get_labels(dataset)
    with open(POI_file) as f:
        sheet = list( csv.reader(f) )
    POI = np.array(sheet[1:])[:,2:5].astype(float)
    
    return dict(zip(labels, torch.tensor(POI, dtype=torch.float32)))

    
def label_to_data_object_map(dataset):
    labels = get_labels(dataset)
    
    return {l: label_to_data_object(dataset, l) for l in labels}

def get_dataloader(dataset, **kwargs):
    pois = label_to_poi_map(dataset)
    datas = label_to_data_object_map(dataset)
    
    for l, d in datas.items():
        d.y = pois[l]
        
    return DataLoader(list(datas.values()), **kwargs)