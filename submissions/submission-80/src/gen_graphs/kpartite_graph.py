import random
from munch import Munch
import json
import os


config = {
    "Type": "KPartiteGraph_Bernoulli",
    "K": 7,
    "N": 2500,
    "edge_probability": 0.001,
    "directed": True,
    "random_seed": 0,
}

def gen_graph_kpartite(config):
    config = Munch(config)
    random.seed(config.random_seed)

    assert config.Type == "KPartiteGraph_Bernoulli"
    K = config.K
    N = config.N
    edge_probability = config.edge_probability
    Edge_list = []
    edge_set = set()

    for k in range(K-1):
        total_edge = int(N * N * edge_probability)
        for _ in range(total_edge):
            i = random.randint(1, N)
            j = random.randint(1, N)
            u = i + k * N
            v = j + (k+1) * N
            while (u, v) in edge_set:
                i = random.randint(1, N)
                j = random.randint(1, N)
                u = i + k * N
                v = j + (k+1) * N
            edge_set.add((u, v))
            Edge_list.append((u, v))
    output_dir = "graphs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{config.Type}_{K}_{N}_{edge_probability}_rs{config.random_seed}.json")

    data = {
            "config": config.toDict(),
            "Edge_list": Edge_list
        }


    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Graph saved to {output_path}")