# First version: 19th of March 2022
# Author: Felix Herter, Nikolas Tapia
# Copyright 2022 Weierstrass Institute
# Copyright 2022 Zuse Institute Berlin
# 
#    This software was developed during the Math+ "Maths meets Image" hackathon 2022.
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import pyvista as pv
from pathlib import Path
import csv
import math
import re
import numpy as np
from collections import defaultdict

plys_to_poi_filename = { 'Bohunician': 'Bohunician', 'PunchedBlade': 'PunchedBlade', 'Shea_Levallois': 'SheaLevallois'}
base_path = Path('/mnt/materials/SIRF/MathPlusBerlin/DATA/LihicFlakesPOI/')

def mesh_to_data_object(mesh, downsample=0.95): #make `downsample` visible from the outside
    mesh = mesh.decimate(downsample)
    pos = torch.tensor(mesh.points, dtype=torch.float32)
    face = torch.tensor(mesh.faces.reshape(-1,4)[:,1:].T, dtype=torch.long)
    norm = torch.tensor(mesh.point_normals, dtype=torch.float32)
    
    transform = T.FaceToEdge()
    graph = Data(pos=pos, face=face, norm=norm)
    return transform(graph)

def generate_targets(graph, poi, std=1):
    graph.y = torch.exp( -torch.linalg.norm(poi - graph.pos, axis=1) ** 2 / (2 * std) )
    graph.y = graph.y / graph.y.sum()
    graph.x = torch.ones((graph.y.shape[0],1))
    graph.y = torch.ones((graph.y.shape[0],1)) / graph.y.sum()
    
    return graph

def label_to_data_object(dataset, label):
    filename = label_to_filename_map(dataset)[label][0]
    mesh = pv.read(base_path / f'3DModels_plys/{dataset}_plys/{filename}')
    data = mesh_to_data_object(mesh)
    poi = label_to_poi_map(dataset)[label]
    generate_targets(data, poi)
    
    return data
    
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

def get_data_list(dataset):
    pois = label_to_poi_map(dataset)
    datas = label_to_data_object_map(dataset)
    
    return list(datas.values())

# Dummy Data generation 

def fibonacci_sphere(samples=1000):
    # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return np.asarray(points)

def generate_cone_meshes(num_cones=100):
    directions = fibonacci_sphere(num_cones)
    center = np.array((0,0,0))
    radius = 0.2
    height = 1

    cones = [pv.Cone(center=center, direction=d, height=height, radius=radius) for d in directions]
#    cones2 = [pv.Cone(center=center-d * height * 1.3/2, direction=-d, height=0.3 * height, radius=radius) for d in directions]
 #   cones = []
    for (cone, d) in zip(cones, directions):
        cone.points = np.vstack((cone.points, -d * height * 0.8))
        new_faces = cone.faces[7:]
        cone.faces = np.concatenate([new_faces, np.array([3, 7, 2, 1, 3, 7, 3, 2, 3, 7, 4, 3, 3, 7, 5, 4, 3, 7, 6, 5, 3, 7, 1, 6])])
        
    apexes = [center + d * height / 2 for d in directions]

    return [cones, apexes]
            
def triangulate_face(face):
    assert(face[0] >= 3)
    if face[0] == 3:
        return np.array(face)
    result = []
    a = face[1]
    for b, c in zip (face[2:], face[3:]):
        result += [3, a, b, c]
    return np.asarray(result)

def triangulate_faces(faces):
    triangulated = []
    i = 0
    while i < len(faces):
        face = faces[i:i+faces[i]+1]
        triangulated.append(triangulate_face(face))
        i += faces[i] + 1
    return np.concatenate(triangulated)

def generate_cone_data_objects(num_cones=100):
    cones, apexes = generate_cone_meshes(num_cones)
    
    for cone in cones:
        cone.faces = triangulate_faces(cone.faces)
    
    result = []
    for cone, apex in zip(cones, apexes):
        mesh = mesh_to_data_object(cone, 0)
        mesh.x = torch.ones((cone.points.shape[0], 1)) / cone.points.shape[0]
        mesh.y = torch.tensor([1] + [0] * (mesh.pos.shape[0] - 1), dtype=torch.float32)
        result.append(mesh)
    return result