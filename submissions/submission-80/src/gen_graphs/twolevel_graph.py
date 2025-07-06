import random
from munch import Munch
import json
import os

def prufer_to_tree(prufer, Node_list):
    n = len(prufer) + 2  
    if len (Node_list) <= 1:
        return []
    degree = [1] * n  
    for node in prufer:
        degree[node] += 1
    edges = []
    leafs = [i for i in range(n) if degree[i] == 1]
    leafs.sort()

    for node in prufer:
        leaf = leafs.pop(0)
        edges.append((Node_list[leaf], Node_list[node]))
        degree[leaf] -= 1
        degree[node] -= 1
        
        if degree[node] == 1:
            leafs.append(node)
            
    leaf1, leaf2 = leafs
    edges.append((Node_list[leaf1], Node_list[leaf2]))
    return edges

def gen_graph_twolevel(config):
    config = Munch(config)
    random.seed(config.random_seed)

    if config.Type == "Two-levelGraph":
        N1 = config.N1
        N2 = config.N2
        N = N1 * N2
        graph_type1 = config.graph_type1
        graph_type2 = config.graph_type2

        Edge_list = []
        for i in range(1, N+1, N2):
            if graph_type2 == "TAE":
                prufer_seq = [random.randint(0, N2-1) for _ in range(N2-2)]
                Tree_Edge_list = prufer_to_tree(prufer_seq, Node_list = list(range(i, i+N2)))
                Current_Edge_list = Tree_Edge_list
                num_additional_edges = int(config.additional_edge_probability * N2 * (N2-1) / 2)
                for _ in range(num_additional_edges):
                    u = random.randint(i, i+N2-1)
                    v = random.randint(i, i+N2-1)
                    if u != v and (u, v) not in Current_Edge_list and (v, u) not in Current_Edge_list:
                        Current_Edge_list.append((u, v))
                Edge_list += Current_Edge_list
            else:
                raise ValueError("Invalid subgraph type.")
        
        if graph_type1 == "Clique":
            Outer_Edge_list = []
            for i in range(1, N1+1):
                for j in range(i+1, N1+1):
                    Outer_Edge_list.append((random.randint((i-1)*N2+1, i*N2), random.randint((j-1)*N2+1, j*N2)))
            Edge_list += Outer_Edge_list
        elif graph_type1 == "TAE":
            assert config.upper_directed == False
            prufer_seq = [random.randint(0, N1-1) for _ in range(N1-2)]
            Outer_Cluster_Edge_list = prufer_to_tree(prufer_seq, Node_list = list(range(0, N1)))
            Outer_Edge_list = []
            for i in range(len(Outer_Cluster_Edge_list)):
                Outer_Edge_list.append((random.randint(Outer_Cluster_Edge_list[i][0]*N2+1, (Outer_Cluster_Edge_list[i][0]+1)*N2), random.randint(Outer_Cluster_Edge_list[i][1]*N2+1, (Outer_Cluster_Edge_list[i][1]+1)*N2)))
            
            num_additional_edges = int(config.upper_edge_probability * N1 * (N1-1) / 2)
            for _ in range(num_additional_edges):
                u = random.randint(0, N1-1)
                v = random.randint(0, N1-1)
                if u != v and (u, v) not in Outer_Cluster_Edge_list and (v, u) not in Outer_Cluster_Edge_list:
                    Outer_Cluster_Edge_list.append((u,v))
                    Outer_Edge_list.append((random.randint(u*N2+1, (u+1)*N2), random.randint(v*N2+1, (v+1)*N2)))
            Edge_list += Outer_Edge_list
        else:
            raise ValueError("Invalid subgraph type.")

        output_dir = "graphs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{config.Type}_{config.graph_type1}{'_' + str(config.upper_edge_probability) if config.graph_type1 in ['TAE'] else ''}_{config.graph_type2}{'_'+str(config.additional_edge_probability) if 'additional_edge_probability' in config else ''}_{config.N1}_{config.N2}.json")

        data = {
            "config": config.toDict(),
            "Edge_list": Edge_list,
            "Outer_Edge_list": Outer_Edge_list,
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Graph saved to {output_path}")

    else:
        raise ValueError("Invalid graph type.")
