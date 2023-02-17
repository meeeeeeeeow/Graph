import numpy as np
import math
import networkx as nx

from shapely.geometry import Polygon
from shapely.affinity import scale

from deepsvg.svglib.svg import SVG
from deepsvg.difflib.tensor import SVGTensor

def build_graph(file_path):    
    # sample contour
    points, colors = sample_points(file_path)
    
    # graph init (add nodes)
    num = 0
    for p in points:
        num += len(p)
    G_point, G_group, node_group, nodes = add_nodes(num, points, colors)
    
    # add edges for each node
    edge_type = {}  # {(n1, n2): type of connection}
    for k, v in node_group.items():
        for i in range(len(v)-1):
            G_point.add_edge(v[i], v[i+1])
            G_point.add_edge(v[i], v[i])  # self
            
            edge_type[(v[i], v[i+1])] = "group"
            edge_type[(v[i+1], v[i])] = "group"
            edge_type[(v[i], v[i])] = "self"
        
        G_point.add_edge(v[-1], v[-1])  # self
        edge_type[(v[-1], v[-1])] = "self"
        
    # merge cluster
    # check mirror or symmetric
    temp_pair = {}
    merge_pair = {}
    for n1 in range(len(nodes)):
        for n2 in range(n1+1, len(nodes)):
            shape1 = G_group.nodes[n1]['features']['pos']
            shape2 = G_group.nodes[n2]['features']['pos']
            temp_pair = check_symmetric(temp_pair, n1, n2, shape1, shape2, G_group)
    
    for k, v in temp_pair.items():
        not_append = True
        n1, n2 = min(k), max(k)
        for kk, vv in merge_pair.items():
            if n1 in kk and n2 in kk:
                not_append = False
                break
            if n1 in kk:
                new_key = list(kk)
                new_key.append(n2)
                merge_pair[tuple(new_key)] = np.array(vv)
                del merge_pair[kk]
                not_append = False
                break
        if not_append:
            merge_pair[(n1, n2)] = np.array(G_group.nodes[k[0]]['features']['rgb'])

    G_group, node_group = merge_cluster(merge_pair, G_point, G_group, node_group)

    # check color and min distance
    temp_pair = {}
    merge_pair = {}
    for n1 in range(len(G_group.nodes)):
        for n2 in range(n1+1, len(G_group.nodes)):
            rgb1 = G_group.nodes[n1]['features']['rgb']
            rgb2 = G_group.nodes[n2]['features']['rgb']
            shape1 = G_group.nodes[n1]['features']['pos']
            shape2 = G_group.nodes[n2]['features']['pos']
            if (rgb1==rgb2).all():
                temp_pair = check_distance(temp_pair, n1, n2, shape1, shape2, rgb1)
                
    for k, v in temp_pair.items():
        not_append = True
        n1, n2 = min(k), max(k)
        for kk, vv in merge_pair.items():
            if n1 in kk and n2 in kk:
                not_append = False
                break
            if n1 in kk:
                new_key = list(kk)
                new_key.append(n2)
                merge_pair[tuple(new_key)] = np.array(vv)
                del merge_pair[kk]
                not_append = False
                break
        if not_append:
            merge_pair[(n1, n2)] = np.array(G_group.nodes[k[0]]['features']['rgb'])
    G_group, node_group = merge_cluster(merge_pair, G_point, G_group, node_group)
    
    # check edges between clusters
    group_edges = {}  
    for n1 in range(len(G_group.nodes)):
        for n2 in range(n1+1, len(G_group.nodes)):
            shape1 = G_group.nodes[n1]['features']['pos']
            shape2 = G_group.nodes[n2]['features']['pos']
            group_edges = block_intersect(n1, n2, shape1, shape2, group_edges)
    
    # add edges between clusters
    for k, v in group_edges.items():
        for n1 in node_group[k[0]]:
            for n2 in node_group[k[1]]:
                G_point.add_edge(n1, n2)
                edge_type[(n1, n2)] = v
                edge_type[(n2, n1)] = v
                
    return G_point, edge_type

def block_intersect(n1, n2, shape1, shape2, group_edges):
    poly1 = Polygon(shape1.tolist()).buffer(0.01)
    poly2 = Polygon(shape2.tolist()).buffer(0.01)
    a1 = poly1.area
    a2 = poly2.area
    intersect = poly1.intersection(poly2).area
    
    if abs(a1-intersect) < 1 or abs(a2-intersect) < 1:  # in / contain
        group_edges[(n1, n2)] = "contain"
    elif intersect != 0:  # overlap
        group_edges[(n1, n2)] = "overlap"
        
    return group_edges

# check mirror or symmetric
def check_symmetric(temp_pair, n1, n2, shape1, shape2, G_group):    
    # flip the blocks
    mean1 = np.mean(shape1 ,axis=0)[0]
    mean2 = np.mean(shape2 ,axis=0)[0]
    flip_x = (mean1 + mean2) / 2
    poly1 = Polygon(shape1.tolist())
    poly2 = Polygon(shape2.tolist())
    if mean1 < mean2:  # shape1 is on the left
        left = scale(poly1, xfact=-1, origin=(flip_x, 0)).buffer(0.01)  # to avoid self-intersection
        right = poly2.buffer(0.01)
        new_color = G_group.nodes[n1]['features']['rgb']
    else:  # shape2 is on the left
        left = scale(poly2, xfact=-1, origin=(flip_x, 0)).buffer(0.01)
        right = poly1.buffer(0.01)
        new_color = G_group.nodes[n2]['features']['rgb']
        
    # calculate iou
    intersect = left.intersection(right).area
    union = left.union(right).area
    iou = intersect / union   
    if iou > 0.6:
        temp_pair[(n1, n2)] = np.array(new_color)
        
    return temp_pair
    
# check color and min distance
def check_distance(temp_pair, n1, n2, shape1, shape2, rgb):
    min_dis = float('inf')
    for i in range(len(shape2)):
        dis = math.sqrt(np.sum((shape2[i]-shape1)**2, axis=1).min())
        if dis < min_dis:
            min_dis = dis
    if min_dis < 2:
        temp_pair[(n1, n2)] = np.array(rgb)
            
    return temp_pair
    
# merge
def merge_cluster(merge_pair, G_point, G_group, node_group):
    G = nx.Graph()
    new_node_group = {}
    cluster_id = 0
    done_list = []  # store already merge cluster idx
    
    for pos, rgb in merge_pair.items():
        new_node_group[cluster_id] = []
        for n in pos:
            done_list.append(n)
            new_node_group[cluster_id] += node_group[n]  # update node_group
        
        # update G_group
        temp = []
        for i in new_node_group[cluster_id]:
            G_point.nodes[i]['features']['rgb'] = rgb  # update point ground truth color
            temp.append(G_point.nodes[i]['features']['pos'])
        features = {
            "pos": np.array(temp),
            "rgb": rgb,
        }
        G.add_node(cluster_id, features=features)
        cluster_id += 1          
        
    # not merged blocks
    for i in G_group.nodes:
        if i not in done_list:
            new_node_group[cluster_id] = node_group[i]  # update node_group
            done_list.append(i)  # update done_list
            
            features = {
                "pos": G_group.nodes[i]['features']['pos'],
                "rgb": np.array(G_group.nodes[i]['features']['rgb'])
            }
            G.add_node(cluster_id, features=features)
            cluster_id += 1
            
    return G, new_node_group

def add_nodes(num, points, colors):
    nodes = {
        node: {
            "pos": []
        } for node in range(num)
    }
    
    G_point = nx.Graph()  # set each point as a node
    G_group = nx.Graph()  # for checking edges between clusters
    node_group = {}  # {cluster idx: [node group idx]}
    node_id = 0  # iter node idx
    cluster_id = 0  # iter cluster idx
    
    for i, groups in enumerate(points):
        h = colors[i].lstrip('#')
        if len(h) == 3: h *= 2
        rgb = list(int(h[i:i+2], 16)/255 for i in (0, 2, 4))
        
        for group in groups:
            temp = []  # store node idx in the same group       
            for p in group: 
                features = {
                    "rgb": np.array(rgb),
                    "pos": np.array(p),
                    "origin_cluster": cluster_id,
                }
                
                nodes[cluster_id]["pos"].append(p.cpu().tolist())
                G_point.add_node(node_id, features=features)
                temp.append(node_id)
                node_id += 1
            
            nodes[cluster_id]["rgb"] = rgb 
            node_group[cluster_id] = temp
            cluster_id += 1
            
    # build G_group
    for node in nodes:
        features = {
            "rgb": nodes[node]["rgb"],
            "pos": np.array(np.array(nodes[node]["pos"]))
        }
        G_group.add_node(node, features=features)
    
    return G_point, G_group, node_group, nodes

def sample_points(file_path):
    # parsing SVG
    svg = SVG.load_svg(file_path).normalize().zoom(0.9)
    svg_target, group_svg_target = svg.to_tensor()
    new_group = []
    colors = []
    for i, g in enumerate(group_svg_target):
        new_group.append(SVGTensor.from_data(g))
        colors.append(svg.colors[i])

    # sample points
    points = []
    for i, g in enumerate(new_group):
        p_target = g.sample_points(n=5)
        points.append(p_target)
        
    return points, colors