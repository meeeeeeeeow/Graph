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
    G_close, nodes = add_nodes(num, points, colors)
    
    # add self edges
    edge_type = {}  # {(n1, n2): type of connection}
    for n in G_close.nodes:
        G_close.add_edge(n, n)
        edge_type[(n, n)] = "self"
        
    # merge cluster
    # check mirror or symmetric
    temp_pair = {}
    merge_pair = {}
    for n1 in range(len(nodes)):
        for n2 in range(n1+1, len(nodes)):
            shape1 = G_close.nodes[n1]['features']['v']
            shape2 = G_close.nodes[n2]['features']['v']
            temp_pair = check_symmetric(temp_pair, n1, n2, shape1, shape2, G_close)

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
                new_pos = np.array((vv[0] + G_close.nodes[n1]['features']['pos']) / 2)
                new_rgb = np.array(vv[1])
                merge_pair[tuple(new_key)] = [new_pos, new_rgb]
                del merge_pair[kk]
                not_append = False
                break
        if not_append:
            new_pos = np.array((G_close.nodes[n1]['features']['pos'] + G_close.nodes[n2]['features']['pos']) / 2)
            new_rgb = np.array(G_close.nodes[n1]['features']['rgb'])
            merge_pair[(n1, n2)] = [new_pos, new_rgb]
    G_close = merge_cluster(merge_pair, G_close)

    # check color and min distance
    temp_pair = {}
    merge_pair = {}
    for n1 in range(len(G_close.nodes)):
        for n2 in range(n1+1, len(G_close.nodes)):
            rgb1 = G_close.nodes[n1]['features']['rgb']
            rgb2 = G_close.nodes[n2]['features']['rgb']
            shape1 = G_close.nodes[n1]['features']['v']
            shape2 = G_close.nodes[n2]['features']['v']
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
                new_pos = np.array((vv[0] + G_close.nodes[n1]['features']['pos']) / 2)
                new_rgb = np.array(vv[1])
                merge_pair[tuple(new_key)] = [new_pos, new_rgb]
                del merge_pair[kk]
                not_append = False
                break
        if not_append:
            new_pos = np.array((G_close.nodes[n1]['features']['pos'] + G_close.nodes[n2]['features']['pos']) / 2)
            new_rgb = np.array(G_close.nodes[n1]['features']['rgb'])
            merge_pair[(n1, n2)] = [new_pos, new_rgb]
    G_close = merge_cluster(merge_pair, G_close)
    
    # check edges between clusters
    group_edges = {}
    for n1 in range(len(G_close.nodes)):
        for n2 in range(n1+1, len(G_close.nodes)):
            shape1 = G_close.nodes[n1]['features']['v']
            shape2 = G_close.nodes[n2]['features']['v']
            group_edges = block_intersect(n1, n2, shape1, shape2, group_edges)

    # add edges
    edge_type = {}
    for k, v in group_edges.items():
        G_close.add_edge(k[0], k[1])
        edge_type[(k[0], k[1])] = v
        
    for n in G_close.nodes:
        G_close.add_edge(n, n)
        edge_type[(n, n)] = "self"
                
    return G_close, edge_type

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
def check_symmetric(temp_pair, n1, n2, shape1, shape2, G_close):    
    # flip the blocks
    mean1 = np.mean(shape1 ,axis=0)[0]
    mean2 = np.mean(shape2 ,axis=0)[0]
    flip_x = (mean1 + mean2) / 2
    poly1 = Polygon(shape1.tolist())
    poly2 = Polygon(shape2.tolist())
    if mean1 < mean2:  # shape1 is on the left
        left = scale(poly1, xfact=-1, origin=(flip_x, 0)).buffer(0.01)  # to avoid self-intersection
        right = poly2.buffer(0.01)
        new_color = G_close.nodes[n1]['features']['rgb']
    else:  # shape2 is on the left
        left = scale(poly2, xfact=-1, origin=(flip_x, 0)).buffer(0.01)
        right = poly1.buffer(0.01)
        new_color = G_close.nodes[n2]['features']['rgb']
        
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
def merge_cluster(merge_pair, G_close):
    G = nx.Graph()
    cluster_id = 0
    done_list = []  # store already merge cluster idx
    
    for points, (pos, rgb) in merge_pair.items():
        new_pos = np.array(pos)
        new_rgb = np.array(rgb)
        new_v = np.array([[-1, -1]])
        new_e = np.array([[-1, -1]])
        node_cnt = 0
        
        for n in points:
            done_list.append(n)
            new_v = np.append(new_v, G_close.nodes[n]['features']['v'], axis=0)
            new_e = np.append(new_e, G_close.nodes[n]['features']['e'] + node_cnt, axis=0)  # reorder cluster's node index
            node_cnt += len(G_close.nodes[n]['features']['v'])
        new_v = np.delete(new_v, 0, axis=0)
        new_e = np.delete(new_e, 0, axis=0)
        
        features = {
            "pos": new_pos,
            "rgb": new_rgb,
            "v": np.array(new_v),
            "e": np.array(new_e)
        }
        G.add_node(cluster_id, features=features)
        cluster_id += 1   
        
    # not merged blocks
    for i in G_close.nodes:
        if i not in done_list:
            done_list.append(i)  # update done_list
            
            features = {
                "pos": np.array(G_close.nodes[i]['features']['pos']),
                "rgb": np.array(G_close.nodes[i]['features']['rgb']),
                "v": np.array(G_close.nodes[i]['features']['v']),
                "e": np.array(G_close.nodes[i]['features']['e'])
            }
            G.add_node(cluster_id, features=features)
            cluster_id += 1
            
    return G

def add_nodes(num, points, colors):
    nodes = {
        node: {
            "rgb": [],
            "pos": []
        } for node in range(num)
    }
    
    # init graph nodes
    G_close = nx.Graph()
    group_idx = []
    idx = 0
    for i, groups in enumerate(points):
        # hex to rgb
        h = colors[i].lstrip('#')
        if len(h) == 3: h *= 2
        rgb = list(int(h[i:i+2], 16)/255 for i in (0, 2, 4))
            
        for group in groups:
            temp = []
            for j, p in enumerate(group):
                nodes[idx]["rgb"].append(np.array(rgb))
                nodes[idx]["pos"].append(np.array(p))
                temp.append(j)
            idx += 1
            group_idx.append(temp)
            
    # build G_group
    for node in nodes:
        # subgraph (add nodes)
        G_temp = nx.Graph()
        for n, i in zip(nodes[node]["pos"], group_idx[node]):
            G_temp.add_node(i, features=n)
            
        # subgraph (add edges)
        for i in range(len(group_idx[node])-1):
            G_temp.add_edge(group_idx[node][i], group_idx[node][i+1])
            G_temp.add_edge(group_idx[node][i], group_idx[node][i])
        G_temp.add_edge(group_idx[node][-1], group_idx[node][-1])
    
        m = len(G_temp.edges)
        edges = np.zeros([2*m,2]).astype(np.int64)
        for e,(s,t) in enumerate(G_temp.edges):
            edges[e, 0] = s
            edges[e, 1] = t
            edges[m+e, 0] = t
            edges[m+e, 1] = s
            
        # main graph
        rgb_mean = np.mean(nodes[node]["rgb"], axis=0)
        pos_mean = np.mean(nodes[node]["pos"], axis=0)
        features = {
            "rgb": rgb_mean,
            "pos": pos_mean,
            "v": np.array(nodes[node]["pos"]),
            "e": np.array(edges)
        }
        G_close.add_node(node, features=features)
    
    return G_close, nodes

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