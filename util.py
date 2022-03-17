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

def mesh_to_data(mesh):
    pos = torch.tensor(mesh.points, dtype=torch.float32)
    face = torch.tensor(mesh.faces.reshape(-1,4)[:,1:].T, dtype=torch.long)
    norm = torch.tensor(mesh.point_normals, dtype=torch.float32)
    
    transform = T.Compose([T.FaceToEdge(remove_faces=False), T.PointPairFeatures()])
    graph = Data(pos=pos, face=face, norm=norm)
    
    return transform(graph)

def label_to_filename_map(dataset):
    poi = plys_to_poi_filename[dataset]
    ply_path = Path(f'/mnt/materials/SIRF/MathPlusBerlin/DATA/LihicFlakesPOI/3DModels_plys/{dataset}_plys/')
    POI_file = Path(f'/mnt/materials/SIRF/MathPlusBerlin/DATA/LihicFlakesPOI/POIs_csv/{poi}_POIs_Point.csv')
    with open(POI_file) as f:
        sheet = list( csv.reader(f) )
    labels = [re.split(r'_|-|\s', ln[1])[0] for ln in sheet[1:]]
    POI = np.array(sheet[1:])[:,2:5].astype(float)
    
    label_int = {l: int(l[1:]) for l in labels}
    label_int_to_filename = defaultdict(list)
    for p in ply_path.glob('*.ply'):
        if len(re.findall(r'\d+', p.name)) > 0:
            label_int_to_filename[int(re.findall(r'\d+', p.name)[0])].append(p.name)
            
    return {l: label_int_to_filename[label_int[l]] for l in labels}
    