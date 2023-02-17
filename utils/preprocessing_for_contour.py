import numpy as np
import bpy
import torch
import colorsys
import math
import skimage

from shapely.geometry import Polygon
from torch_geometric.data import Data

# check overlap
def block_intersect(i, j, poly1, poly2, group_edges, group_edges_type):
    a1 = poly1.area
    a2 = poly2.area
    intersect = poly1.intersection(poly2).area
    dis = poly1.distance(poly2)
    
    if abs(intersect - min(a1, a2)) < 0.000001:  # contain
        pass
    elif intersect != 0:  # overlap
        intersect = 1 / (intersect * 100)
        group_edges.append([i, j])
        group_edges.append([j, i])
        group_edges_type.append([intersect])
        group_edges_type.append([intersect])
    
    # if abs(intersect - min(a1, a2)) < 0.000001:  # in
    #     group_edges.append([i, j])
    #     group_edges.append([j, i])
    #     group_edges_type.append(2)
    #     group_edges_type.append(2)
    # elif intersect != 0:  # overlap
    #     group_edges.append([i, j])
    #     group_edges.append([j, i])
    #     group_edges_type.append(1)
    #     group_edges_type.append(1)
        
    return group_edges, group_edges_type

def create_data(file_path):
    # clean the scene and import SVG file
    bpy.ops.object.select_all()
    bpy.ops.object.delete()
    bpy.ops.import_curve.svg(filepath=file_path)
    
    # rescale
    curve_objs = [ob for ob in bpy.data.objects if ob.type == "CURVE"]
    bpy.ops.object.select_all(action="DESELECT")
    for ob in curve_objs:
        ob.select_set(True)
    bbox = [ob.bound_box for ob in curve_objs]
    x_size = max(p[0] for b in bbox for p in b) - min(p[0] for b in bbox for p in b)
    y_size = max(p[1] for b in bbox for p in b) - min(p[1] for b in bbox for p in b)
    scale = 0.1 / max(x_size, y_size)
    bpy.ops.transform.resize(value=(scale, scale, scale))
    bpy.ops.object.select_all(action="DESELECT")
    
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
    hsv = {}
    lab = {}
    idx = 0
    for ob in bpy.data.objects:
        if ob.type == "MESH" and "mesh_obj" in ob.name:
            try:
                # get mesh
                rgb = ob.material_slots[0].material.diffuse_color
                colors[idx] = np.array([rgb[0], rgb[1], rgb[2]])
                hsv[idx] = np.array(colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2]))
                l, a, b = np.array(skimage.color.rgb2lab([rgb[0], rgb[1], rgb[2]]))
                l, a, b = l/100, (a+128)/256, (b+128)/256
                lab[idx] = [l, a, b]
                
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
            
    # get polygon
    poly_nodes = {}
    all_polys = []
    group_x = []
    idx = 0
    for i, points in nodes.items():
        if len(points) < 4: continue
        poly = Polygon(points).buffer(0.001)
        all_polys.append(poly)
        
        # w/ simplification
        sim = poly.simplify(0.0001, preserve_topology=True)
        poly_nodes[idx] = []
        if sim.geom_type == "MultiPolygon":
            coords_list = [list(x.exterior.coords) for x in sim.geoms]
            for c in coords_list:
                for x, y in c:
                    poly_nodes[idx].append([x, y])
        else:
            xx, yy = sim.exterior.coords.xy
            for x, y in zip(xx, yy):
                poly_nodes[idx].append([x, y])
        
        # # w/o simplification
        # poly_nodes[idx] = []
        # if poly.geom_type == "MultiPolygon":
        #     coords_list = [list(x.exterior.coords) for x in poly.geoms]
        #     for c in coords_list:
        #         for x, y in c:
        #             poly_nodes[idx].append([x, y])
        # else:
        #     xx, yy = poly.exterior.coords.xy
        #     for x, y in zip(xx, yy):
        #         poly_nodes[idx].append([x, y])
                
        # center of closed curve
        x_mean = np.mean(np.array(poly_nodes[idx])[:, 0])
        y_mean = np.mean(np.array(poly_nodes[idx])[:, 1])
        group_x.append([x_mean, y_mean])
        idx += 1
        
    if len(poly_nodes) == 0:
        return Data()
        
    # get nodes and edges
    all_points = []
    all_edges = []
    all_rgb = []
    all_hsv = []
    all_lab = []
    all_cluster = []
    idx = 0
    for i, p in poly_nodes.items():
        cnt = 0
        first_idx = idx
        for j in range(len(p)-1):
            all_points.append(p[j])
            if j < len(p)-2: all_edges.append([idx, idx+1])
            else: all_edges.append([idx, first_idx])
            idx += 1
            cnt += 1
        
        all_cluster += [i for _ in range(cnt)]
        
        rgb_temp = [colors[i].tolist()] * cnt
        hsv_temp = [hsv[i].tolist()] * cnt
        lab_temp = [lab[i]] * cnt
        if i == 0:
            all_rgb = rgb_temp
            all_hsv = hsv_temp
            all_lab = lab_temp
        else:
            all_rgb = np.concatenate((all_rgb, rgb_temp), axis=0)
            all_hsv = np.concatenate((all_hsv, hsv_temp), axis=0)
            all_lab = np.concatenate((all_lab, lab_temp), axis=0)
    all_points = np.array(all_points)
    all_edges = np.array(all_edges)
    all_rgb = np.array(all_rgb)
    all_hsv = np.array(all_hsv)
    all_lab = np.array(all_lab)
    all_cluster = np.array(all_cluster, dtype=np.int64)
    all_h = (np.round(all_hsv[:,0]*10)%10).astype(np.int64)
    group_x = np.array(group_x)

    # check group connection
    group_edges = []
    group_edges_type = []
    # for i in range(len(all_polys)-1):
    #     for j in range(i+1, len(all_polys)):
    #         group_edges, group_edges_type = block_intersect(i, j, all_polys[i], all_polys[j], group_edges, group_edges_type)
    
    # create pyg data
    y_rgb = torch.Tensor(all_rgb)
    y_hsv = torch.Tensor(all_hsv)
    y_lab = torch.Tensor(all_lab)
    y_h = torch.LongTensor(all_h)
    y_cluster = torch.Tensor(all_cluster).type(torch.long)

    # nodes
    # x = torch.Tensor(all_points)
    n = all_points.shape[0]
    x = np.zeros([n, 5]).astype(np.float32)
    for i, points in enumerate(all_points):
        temp = np.concatenate(([0, 0, 0], points))
        x[i] = temp
    x = torch.Tensor(x)

    # edges
    m = all_edges.shape[0]
    edges = np.zeros([2*m, 2]).astype(np.int64)
    edge_attr = np.zeros([2*m, 2]).astype(np.float32)
    for e, (s,t) in enumerate(all_edges):
        edges[e, 0] = s
        edges[e, 1] = t
        edges[m+e, 0] = t
        edges[m+e, 1] = s
        
        (x1, y1), (x2, y2) = all_points[s], all_points[t]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = math.atan2(y2 - y1, x2 - x1)
        edge_attr[e] = [distance, angle]
        edge_attr[m+e] = [distance, angle]
    edges = torch.Tensor(np.transpose(edges)).type(torch.long)
    edge_attr = torch.Tensor(edge_attr)
    
    group_x = torch.Tensor(group_x)
    group_edges = torch.Tensor(np.transpose(group_edges)).type(torch.long)
    # group_edge_attr = torch.Tensor(np.expand_dims(np.array(group_edges_type), axis=1))
    group_edge_attr = torch.Tensor(group_edges_type).type(torch.float)
    
    data = Data(x=x,
                edge_index=edges,
                rgb=y_rgb,
                hsv=y_hsv,
                lab=y_lab,
                h = y_h,
                edge_attr=edge_attr,
                cluster=y_cluster,
                group_x=group_x,
                group_edge_index=group_edges,
                group_edge_attr=group_edge_attr,)
    
    return data