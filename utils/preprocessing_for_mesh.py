import numpy as np
import bpy
import geopandas as gpd
import optimesh
import warnings
import tripy
import torch
import colorsys

from shapely.geometry import Polygon
from shapely.ops import triangulate
from geovoronoi import voronoi_regions_from_coords
from quad_mesh_simplify import simplify_mesh
from torch_geometric.data import Data
from .earclip import triangulate

def create_mesh(file_path):
    # clean the scene and import SVG file
    bpy.ops.object.select_all()
    bpy.ops.object.delete()
    bpy.ops.import_curve.svg(filepath=file_path)
    
    # variables
    colors = {}  # {cluster_id: (r, g, b), ...}
    hsv = {}  # {cluster_id: (h, s, v), ...}
    nodes = {}  # {cluster_id: [x, y] of all vertices in the mesh, ...}
    polys = {}  # {cluster_id: polygon, ...}
    pos_to_idx = {}  # {(x, y): idx, ...}
    idx_to_pos = {}  # {idx: (x, y), ...}
    mesh_points = {}  # {cluster_id: [x, y], ...}
    mesh_faces = {}  # {cluster_id: (n1, n2, n3), ...}
    
    # convert curve to mesh
    idx = 0
    for ob in bpy.data.objects:
        if ob.type == "CURVE":
            mesh = bpy.data.meshes.new_from_object(ob)
            new_obj = bpy.data.objects.new("mesh_obj" + str(idx), mesh)
            new_obj.matrix_world = ob.matrix_world
            bpy.context.collection.objects.link(new_obj)
            idx += 1
            
    # get all meshes
    idx = 0
    for ob in bpy.data.objects:
        if ob.type == "MESH" and "mesh_obj" in ob.name:
            try:
                # get mesh
                rgb = ob.material_slots[0].material.diffuse_color
                colors[idx] = np.array([rgb[0], rgb[1], rgb[2]])
                hsv[idx] = np.array(colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2]))
                
                # get vertices
                v = ob.data.vertices[0]
                coords = [(ob.matrix_world @ v.co) for v in ob.data.vertices]  # (x, y, z)
                nodes[idx] = []
                for x, y, z in coords:
                    nodes[idx].append([x, y]) 
                nodes[idx] = np.array(nodes[idx])
                
                idx += 1
            except:
                continue
        
    # triangulate polygons
    for i, node in nodes.items():
        polys[i] = np.array(triangulate(node.tolist()))
        
    # create mesh points and cells numpy array
    for i, poly_list in polys.items():
        cnt = 0
        pos_to_idx[i] = {}
        idx_to_pos[i] = {}
        mesh_faces[i] = []
        
        for pos in poly_list:
            for p in pos:
                if tuple(p) not in pos_to_idx[i]:
                    pos_to_idx[i][tuple(p)] = cnt
                    cnt += 1
            mesh_faces[i].append([pos_to_idx[i][tuple(pos[0])], pos_to_idx[i][tuple(pos[1])], pos_to_idx[i][tuple(pos[2])]])
        
        idx_to_pos[i] = {v: k for k, v in pos_to_idx[i].items()}
        mesh_faces[i] = np.array(mesh_faces[i])
        
    for i, d in idx_to_pos.items():
        mesh_points[i] = []
        for v in d.values():
            mesh_points[i].append(list(v))
        mesh_points[i] = np.array(mesh_points[i])
        
    # mesh simplification (return new variables)
    new_points = []
    new_colors = []
    new_edges = []
    new_faces = []
    new_cluster = []
    c_idx = 0
    idx = 0
    for i, (ori_points, ori_cells) in enumerate(zip(mesh_points.values(), mesh_faces.values())):
        if len(ori_points) == 0: continue
        
        z = np.zeros([len(ori_points), 1])
        ori_points = np.concatenate((ori_points, z), axis=1)
        ori_cells = ori_cells.astype(np.uint32)
        
        warnings.filterwarnings("ignore")
        
        points, cells = simplify_mesh(ori_points, ori_cells, 3, max_err=0.2)
        points = points[:,:2]
        
        # # mesh optimization
        # points, cells = optimesh.optimize_points_cells(
        #     points, cells, "cpt-quasi-newton", 1.0e-5, 100
        # )
        
        # build points and colors
        # c_temp = [colors[i]] * len(points)
        c_temp = [hsv[i]] * len(points)
        if idx == 0:
            new_points = points
            new_colors = c_temp
            new_faces = cells
        else:
            new_points = np.concatenate((new_points, points), axis=0)
            new_colors = np.concatenate((new_colors, c_temp), axis=0)
            new_faces = np.concatenate((new_faces, cells), axis=0)
        
        # add edges
        for a, b, c in cells:
            new_edges.append([a+idx, b+idx])
            new_edges.append([b+idx, c+idx])
            new_edges.append([c+idx, a+idx])
        idx += len(points)
        
        new_cluster += [c_idx for _ in range(len(points))]
        c_idx += 1
        
    new_points = np.array(new_points)
    new_colors = np.array(new_colors)
    new_edges = np.array(new_edges)
    new_faces = np.array(new_faces)
    new_cluster = np.array(new_cluster)
    
    # create pyg data
    # nodes
    n = new_points.shape[0]
    # x = np.zeros([n, 5]).astype(np.float32)
    x = np.zeros([n, 2]).astype(np.float32)  # (x, y)
    y = np.zeros([n, 3]).astype(np.float32)
    h = np.zeros([n]).astype(np.float32)  # h (s, v)
    for i, (points, colors) in enumerate(zip(new_points, new_colors)):
        y[i, :] = colors
        
        de = colors[0] * 360
        if de <= 30: h[i] = 0
        elif de <= 60: h[i] = 1
        elif de <= 90: h[i] = 2
        elif de <= 120: h[i] = 3
        elif de <= 150: h[i] = 4
        elif de <= 180: h[i] = 5
        elif de <= 210: h[i] = 6
        elif de <= 240: h[i] = 7
        elif de <= 270: h[i] = 8
        elif de <= 300: h[i] = 9
        elif de <= 330: h[i] = 10
        else: h[i] = 11
        
        # f = [0, 0, 0]
        # f += points.tolist()
        # x[i, :] = f
        
        x[i, :] = points.tolist()
        
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    h = torch.Tensor(h)
    new_cluster = torch.LongTensor(new_cluster)

    # edges
    m = new_edges.shape[0]
    edges = np.zeros([2*m, 2]).astype(np.int64)
    edge_attr = np.zeros([2*m, 4]).astype(np.float32)  # (x1, y1, x2, y2)
    for e, (s,t) in enumerate(new_edges):
        edges[e, 0] = s
        edges[e, 1] = t
        edge_attr[e, :2] = new_points[s]
        edge_attr[e, 2:4] = new_points[t]
        
        edges[m+e, 0] = t
        edges[m+e, 1] = s
        edge_attr[m+e, :2] = new_points[t]
        edge_attr[m+e, 2:4] = new_points[s]
        
    edges = torch.Tensor(np.transpose(edges)).type(torch.long)
    edge_attr = torch.Tensor(edge_attr.astype(np.float32))
    new_faces = torch.Tensor(new_faces.astype(np.int64))
    data = Data(x=x, edge_index=edges, y=y, faces=new_faces, h=h, edge_attr=edge_attr, cluster=new_cluster)
    
    return data