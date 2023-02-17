import numpy as np
import bpy
import torch
import skimage
import math

from torch_geometric.data import Data

def create_data(file_path):
    # clean the scene and import SVG file
    bpy.ops.object.select_all()
    bpy.ops.object.delete()
    for curve in bpy.data.curves:
        bpy.data.curves.remove(curve)
    bpy.ops.import_curve.svg(filepath=file_path)
    
    # get bezier curves
    curves = bpy.data.curves
    paths = {
        path: {
            "id": None,
            "rgb": [],
            "path": []
        } for path in range(len(curves))
    }

    for i, curve in enumerate(curves):
        paths[i]['id'] = i
        rgba = curve.materials[0].diffuse_color
        paths[i]['rgb'] = [rgba[0], rgba[1], rgba[2]]

        p = []
        for spline in curve.splines:
            for j, bezier in enumerate(spline.bezier_points):
                start = list(bezier.co[:2])
                control1 = list(bezier.handle_left[:2])
                control2 = list(bezier.handle_right[:2])
                if j < len(spline.bezier_points) - 1:
                    end = list(spline.bezier_points[j+1].co[:2])
                else:
                    end = list(spline.bezier_points[0].co[:2])
                p.append([start, control1, control2, end])
        paths[i]['path'] = np.array(p)
       
    # construct graph 
    x = []
    edge_index = []
    edge_attr = []
    rgb = []
    lab = []
    cluster = []
    idx = 0

    for path in paths.values():
        # rgb, lab, and cluster
        points = path['path']
        cluster += [path['id']] * points.shape[0]
        rgb += [path['rgb']] * points.shape[0]
        l, a, b = np.array(skimage.color.rgb2lab(path['rgb']))
        l, a, b = l/100, (a+128)/256, (b+128)/256
        lab += [[l, a, b]] * points.shape[0]

        # x, edge_index, and edge_attr
        _start = idx
        for i, (start, control1, control2, end) in enumerate(points):
            x.append(start)
            
            if i == len(points)-1:
                edge_index.append([idx, _start])
                edge_index.append([_start, idx])
            else:
                edge_index.append([idx, idx+1])
                edge_index.append([idx+1, idx])
            idx += 1
            
            # edge_attr: [control1_x, control1_y, control2_x, control2_y,
            #             control1_x-start_x, control1_y-start_y, control2_x-end_x,  control2_y-end_y, distance, angle]
            distance = (start[0] - end[0])**2 + (start[1] - end[1])**2
            angle = math.atan2(end[1] - start[1], end[0] - start[0])
            attr = [control1[0], control1[1], control2[0], control2[1]]
            attr += [control1[0]-start[0], control1[1]-start[1], control2[0]-end[0], control2[1]-end[1], distance, angle]
            edge_attr.append(attr)
            angle = math.atan2(start[1] - end[1], start[0] - end[0])
            attr = [control2[0], control2[1], control1[0], control1[1]]
            attr += [control2[0]-end[0], control2[1]-end[1], control1[0]-start[0], control1[1]-start[1], distance, angle]
            edge_attr.append(attr)
        
    _x = torch.Tensor(np.array(x))
    _edge_index = torch.Tensor(np.transpose(np.array(edge_index))).type(torch.long)
    _edge_attr = torch.Tensor(np.array(edge_attr))
    _rgb = torch.Tensor(np.array(rgb))
    _lab = torch.Tensor(np.array(lab))
    _cluster = torch.Tensor(cluster).type(torch.long)

    data = Data(x=_x,
                edge_index=_edge_index,
                edge_attr=_edge_attr,
                rgb=_rgb,
                lab=_lab,
                cluster=_cluster,)

    return data