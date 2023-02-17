import numpy as np
import bpy
import optimesh
import torch
import colorsys

from shapely.geometry import Polygon
from shapely.ops import triangulate
from torch_geometric.data import Data
from scipy.spatial import Delaunay

def create_data(file_path):
    # clean the scene and import SVG file
    bpy.ops.object.select_all()
    bpy.ops.object.delete()
    bpy.ops.import_curve.svg(filepath=file_path)
    
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
    colors = {}
    nodes = {}
    idx = 0
    hsv = {}
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
        
    # triangulate
    polys = {}
    for i, node in nodes.items():
        polys[i] = []
        if len(node) < 4: continue
        
        # tri = Delaunay(node)
        # polys[i] = node[tri.simplices]
        
        poly = Polygon(node).buffer(0.001)
        tri = triangulate(poly)
        for po in tri:
            xx, yy = po.exterior.coords.xy
            temp = []
            for x, y in zip(xx, yy):
                temp.append([x, y])
            polys[i].append(temp[:3])
        polys[i] = np.array(polys[i])
        
    # optim mesh
    all_points = []
    all_edges = []
    all_rgb = []
    all_hsv = []
    all_cluster = []
    add_num = 0
    idx = 0
    for i, poly in polys.items():
        if len(poly) == 0: continue
        
        p_i = 0
        pos2idx = {}
        points = []
        cells = []
        for n1, n2, n3 in poly:
            if tuple(n1) not in pos2idx:
                pos2idx[tuple(n1)] = p_i
                points.append(n1)
                all_rgb += [colors[i].tolist()]
                all_hsv += [hsv[i].tolist()]
                p_i += 1
            if tuple(n2) not in pos2idx:
                pos2idx[tuple(n2)] = p_i
                points.append(n2)
                all_rgb += [colors[i].tolist()]
                all_hsv += [hsv[i].tolist()]
                p_i += 1
            if tuple(n3) not in pos2idx:
                pos2idx[tuple(n3)] = p_i
                points.append(n3)
                all_rgb += [colors[i].tolist()]
                all_hsv += [hsv[i].tolist()]
                p_i += 1       
            cells.append([pos2idx[tuple(n1)], pos2idx[tuple(n2)], pos2idx[tuple(n3)]])
            
        points = np.array(points)
        cells = np.array(cells) 
        points, cells = optimesh.optimize_points_cells(
            points, cells, "cpt-quasi-newton", 1.0e-5, 100
        )
        
        if idx == 0:
            all_points = points
        else:
            all_points = np.concatenate((all_points, points), axis=0)
        for a, b, c in cells:
            all_edges.append([a+add_num, b+add_num])
            all_edges.append([b+add_num, c+add_num])
            all_edges.append([c+add_num, a+add_num])
        add_num += len(points)
        
        all_cluster += [idx for _ in range(len(points))]
        idx += 1
        
    all_points = np.array(all_points)
    all_edges = np.array(all_edges)
    all_rgb = np.array(all_rgb)
    all_hsv = np.array(all_hsv)
    all_cluster = np.array(all_cluster)

    # nodes
    x = torch.Tensor(all_points)
    y_rgb = torch.Tensor(all_rgb)
    y_hsv = torch.Tensor(all_hsv)
    y_cluster = torch.LongTensor(all_cluster)

    # edges
    m = all_edges.shape[0]
    edges = np.zeros([2*m, 2]).astype(np.int64)
    edge_attr = np.zeros([2*m, 4]).astype(np.float32)
    for e, (s,t) in enumerate(all_edges):
        edges[e, 0] = s
        edges[e, 1] = t
        edges[m+e, 0] = t
        edges[m+e, 1] = s
        
        edge_attr[e, :2] = all_points[s]
        edge_attr[e, 2:] = all_points[t]
        edge_attr[m+e, :2] = all_points[t]
        edge_attr[m+e, 2:] = all_points[s]
    edges = torch.Tensor(np.transpose(edges)).type(torch.long)
    edge_attr = torch.Tensor(edge_attr)

    data = Data(x=x, edge_index=edges, rgb=y_rgb, hsv=y_hsv, edge_attr=edge_attr, cluster=y_cluster)
    
    return data