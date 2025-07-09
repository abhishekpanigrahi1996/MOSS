from .utils import encoder, decoder
from collections import deque
import numpy as np
import random
import json
import copy
import os

class  GeneralPathFindingTask:
    def __init__(self, config):
        
        random.seed(config.random_seed)
        self.config = config
        self.rng = random.Random(config.random_seed)
        # breakpoint()
        with open(config.graph_path, "r") as f:
            self.graph_info = json.load(f)
        
        self.task_type = config.task_type
        if self.task_type in ["Two-levelGraph_pt1", "Two-levelGraph_SFT1", "Two-levelGraph_SFT2", "Two-levelGraph_SFT2_planning1", "Two-levelGraph_SFT3", "Two-levelGraph_SFT3_planning2", "Two-levelGraph_onlyplanning2", "Two-levelGraph_SFT4", "Two-levelGraph_SFT4_planning3", "Two-levelGraph_SFT5", "Two-levelGraph_SFT5_planning4", "Two-levelGraph_onlyplanning4", "Two-levelGraph_SFT3_CEL", "Two-levelGraph_pt2"]: # random sample path within each subgraph. # SFT2 is specifically for clique. SFT3: always go through shortest clusters. SFT4: upper-level path is fixed. SFT5: in each cluster, outputs corresponding paths. # CEL: context_enhanced_learning.
            # pt2: a random partition.
            self.prepare_TwolevelGraph()
            if self.task_type == "Two-levelGraph_pt1":
                self.prepare_TwolevelGraph_pt1()
            if self.task_type in ["Two-levelGraph_SFT1", "Two-levelGraph_SFT2", "Two-levelGraph_SFT2_planning1", "Two-levelGraph_SFT3", "Two-levelGraph_SFT3_planning2", "Two-levelGraph_onlyplanning2", "Two-levelGraph_SFT4", "Two-levelGraph_SFT4_planning3", "Two-levelGraph_SFT5", "Two-levelGraph_SFT5_planning4", "Two-levelGraph_onlyplanning4","Two-levelGraph_SFT3_CEL"]:
                if self.task_type in ["Two-levelGraph_onlyplanning2", "Two-levelGraph_onlyplanning4"]:
                    self.pause_token = config.pause_token
                if self.task_type == "Two-levelGraph_SFT3_CEL":
                    self.stage = 0
                
                self.TwolevelGraph_pt1_pairs  = [] # no eval_pairs -- no_eval=True
                for k in range(1, self.N+1, self.N2):
                    for i in range(k, k+self.N2):
                        for j in range(k, k+self.N2):
                            if i != j:
                                self.TwolevelGraph_pt1_pairs.append((i, j))

            if self.task_type == "Two-levelGraph_SFT1":
                self.eval_rate = config.eval_rate    
                self.get_shortest_paths()
                if self.eval_rate >= 0.991:
                    self.prepare_TwolevelGraph_SFT1()
            if self.task_type in ["Two-levelGraph_SFT2" , "Two-levelGraph_SFT2_planning1"]:
                assert self.graph_info["config"]["graph_type1"] == "Clique"
                self.eval_rate = config.eval_rate
                if self.task_type == "Two-levelGraph_SFT2_planning1":
                    self.provide_planning = config.provide_planning
                self.get_shortest_paths_in_cluster()
                self.connecting_edge = [[None for _ in range(self.N1)] for _ in range(self.N1)]
                for (x,y) in self.Outer_Edge_list:
                    cluster_x = (x-1) // self.N2
                    cluster_y = (y-1) // self.N2
                    self.connecting_edge[cluster_x][cluster_y] = (x, y)
                    self.connecting_edge[cluster_y][cluster_x] = (y, x)
                self.prepare_TwolevelGraph_SFT1()
            if self.task_type in ["Two-levelGraph_SFT3", "Two-levelGraph_SFT3_planning2", "Two-levelGraph_onlyplanning2", "Two-levelGraph_SFT5", "Two-levelGraph_SFT5_planning4", "Two-levelGraph_onlyplanning4", "Two-levelGraph_SFT3_CEL"]:
                self.eval_rate = config.eval_rate
                if self.task_type in ["Two-levelGraph_SFT3_planning2", "Two-levelGraph_SFT5_planning4"] :
                    self.provide_planning = config.provide_planning
                if self.task_type in ["Two-levelGraph_SFT5_planning4", "Two-levelGraph_SFT3_planning2"] : # currently only implemented from SFT5_planning4
                    self.random_planning = config.random_planning
                    self.fix_interval = config.fix_interval
                    self.planning_with_cluster_token = config.planning_with_cluster_token
                self.prepare_TwolevelGraph_SFT3()
            elif self.task_type in ["Two-levelGraph_SFT4","Two-levelGraph_SFT4_planning3",]:
                self.eval_rate = config.eval_rate
                if self.task_type == "Two-levelGraph_SFT4_planning3":
                    self.provide_planning = config.provide_planning
                self.prepare_TwolevelGraph_SFT4()
        else:
            raise ValueError("Invalid task type")
    
    def prepare_TwolevelGraph_SFT4(self):
        self.prepare_TwolevelGraph_SFT3()
        self.upperlevel_shortest_path_seq = [[None for _ in range(self.N1)] for _ in range(self.N1)]
        for cluster_S in range(self.N1):
            for cluster_T in range(self.N1):
                if cluster_S != cluster_T:
                    upperlevel_path = [cluster_S]
                    while upperlevel_path[-1] != cluster_T:
                        current_cluster = upperlevel_path[-1]
                        min_dis = min(self.upperlevel_shortest_paths[N][cluster_T] for N in self.adj_list_across_cluster[current_cluster])
                        next_clusters = [node for node in self.adj_list_across_cluster[current_cluster] if self.upperlevel_shortest_paths[node][cluster_T] == min_dis]
                        next_cluster = random.choice(next_clusters)
                        upperlevel_path.append(next_cluster)
                    self.upperlevel_shortest_path_seq[cluster_S][cluster_T] = upperlevel_path


        
    def prepare_TwolevelGraph_SFT3(self):
        self.get_shortest_paths_in_cluster()
        self.connecting_edge = {}
        self.adj_list_across_cluster = [[] for i in range(self.N1)]
        for (x,y) in self.Outer_Edge_list:
            cluster_x = (x-1) // self.N2
            cluster_y = (y-1) // self.N2
            self.connecting_edge[(cluster_x,cluster_y)] = (x, y)
            self.adj_list_across_cluster[cluster_x].append(cluster_y)
            if self.graph_info["config"]["upper_directed"] == False:
                self.connecting_edge[(cluster_y,cluster_x)] = (y, x)
                self.adj_list_across_cluster[cluster_y].append(cluster_x)
        self.get_upperlevel_shortest_paths()
        self.prepare_TwolevelGraph_SFT1()

    def prepare_TwolevelGraph(self):
        self.N1 = self.graph_info["config"]["N1"]
        self.N2 = self.graph_info["config"]["N2"]
        self.N = self.N1 * self.N2
        self.Edge_list = self.graph_info["Edge_list"]
        self.Outer_Edge_list = self.graph_info["Outer_Edge_list"]
        self.adj_list = [[] for i in range(self.N + 1)]
        if self.graph_info["config"]["directed"]:
            for u, v in self.Edge_list:
                self.adj_list[u].append(v)
            self.edge_set = set()
            for u, v in self.Edge_list:
                self.edge_set.add((u, v))
        else: 
            for u, v in self.Edge_list:
                self.adj_list[u].append(v)
                self.adj_list[v].append(u)
            self.edge_set = set()
            for u, v in self.Edge_list:
                self.edge_set.add((u, v))
                self.edge_set.add((v, u))
    
    def prepare_TwolevelGraph_pt1(self):
        self.get_shortest_paths_in_cluster()
        self.shortest_paths = self.shortest_paths_in_cluster
        self.SFT1_train_pairs = [] # no eval_pairs -- no_eval=True
        for k in range(1, self.N+1, self.N2):
            for i in range(k, k+self.N2):
                for j in range(k, k+self.N2):
                    if i != j:
                        self.SFT1_train_pairs.append((i, j))
        self.TwolevelGraph_pt1_pairs = copy.deepcopy(self.SFT1_train_pairs)
    def prepare_TwolevelGraph_SFT1(self):
        if self.eval_rate  >= 0.5 and self.N>=20000: # previous tasks will not be affected.
                
            self.SFT1_train_pairs = []
            train_set = set()
            num_train_set = int(self.N1*(self.N1-1)*self.N2*self.N2*(1-self.eval_rate))

            while len(self.SFT1_train_pairs) < num_train_set:
                k1 = random.randint(0, self.N1-1)
                k2 = random.randint(0, self.N1-1)
                if k1 != k2:
                    i = random.randint(1, self.N2)
                    j = random.randint(1, self.N2)
                    if (i, j) not in train_set:
                        self.SFT1_train_pairs.append((k1*self.N2+i, k2*self.N2+j))
                        train_set.add((k1*self.N2+i, k2*self.N2+j))
            
            for (i,j) in self.SFT1_train_pairs:
                assert (i-1)//self.N2 != (j-1)//self.N2
            self.SFT1_train_pairs_set = set(self.SFT1_train_pairs)
        else:
            self.SFT1_pairs = [] 
            for k1 in range(1, self.N+1, self.N2):
                for k2 in range(1, self.N+1, self.N2):
                    if k1 != k2:
                        for i in range(k1, k1+self.N2):
                            for j in range(k2, k2+self.N2):
                                if i != j:
                                    self.SFT1_pairs.append((i, j))
            random.shuffle(self.SFT1_pairs)
            self.SFT1_eval_pairs = self.SFT1_pairs[:int(len(self.SFT1_pairs) * self.eval_rate)]
            self.SFT1_train_pairs = self.SFT1_pairs[int(len(self.SFT1_pairs) * self.eval_rate)+1:]

    
    def prepare_graph(self):
        if self.graph_info["config"]["Type"] == "Two-levelGraph":
            self.prepare_TwolevelGraph()
            return
        self.N = self.graph_info["config"]["N"]
        self.Edge_list = self.graph_info["Edge_list"]
        self.adj_list = [[] for i in range(self.N + 1)]
        if self.graph_info["config"]["directed"]:
            for u, v in self.Edge_list:
                self.adj_list[u].append(v)
            self.edge_set = set()
            for u, v in self.Edge_list:
                self.edge_set.add((u, v))
        else:
            for u, v in self.Edge_list:
                self.adj_list[u].append(v)
                self.adj_list[v].append(u)
            self.edge_set = set()
            for u, v in self.Edge_list:
                self.edge_set.add((u, v))
                self.edge_set.add((v, u))
    

    def get_mixing_TwolevelGraph_pt1_data(self, pair):
        S, T = pair
        input_id = [encoder(S), encoder(T), encoder(":")]
        label = [-100] * 3
        attention_mask = [1] * 3

        path = [S]
        while path[-1] != T: 
            current_node = path[-1]
            min_dis = min(self.shortest_paths_in_cluster[n][T] for n in self.adj_list_in_cluster[current_node])
            next_nodes = [node for node in self.adj_list_in_cluster[current_node] if self.shortest_paths_in_cluster[node][T] == min_dis]
            next_node = random.choice(next_nodes)
            path.append(next_node)
        
        input_id = input_id + [encoder(node) for node in path] + [encoder("E")]
        label = label + [encoder(node) for node in path] + [encoder("E")]
        attention_mask = attention_mask + [1] * len(path) + [1]
        return input_id, label, attention_mask


    def get_data_TwolevelGraph_SFT4_planning3(self, pair):    
        
        S, T = pair
        input_id = [encoder(S), encoder(T), encoder(":")]
        label = [-100] * 3
        attention_mask = [1] * 3

        cluster_S = (S-1) // self.N2
        cluster_T = (T-1) // self.N2

        upperlevel_path = self.upperlevel_shortest_path_seq[cluster_S][cluster_T]

        path = []
        key_nodes = []
        current_node = S
        for i in range(len(upperlevel_path)-1):
            (M1, M2) = self.connecting_edge[(upperlevel_path[i],upperlevel_path[i+1])]
            path.append(current_node)
            while path[-1] != M1: # the shortest_path is only within cluster!!
                current_node = path[-1]
                min_dis = min(self.shortest_paths_in_cluster[n][M1] for n in self.adj_list_in_cluster[current_node])
                next_nodes = [node for node in self.adj_list_in_cluster[current_node] if self.shortest_paths_in_cluster[node][M1] == min_dis]
                next_node = random.choice(next_nodes)
                path.append(next_node)
            current_node = M2
            # breakpoint()
            key_nodes.append(M1)
            key_nodes.append(M2)
        
        path.append(current_node)
        while path[-1] != T:
            current_node = path[-1]
            min_dis = min(self.shortest_paths_in_cluster[n][T] for n in self.adj_list_in_cluster[current_node])
            next_nodes = [node for node in self.adj_list_in_cluster[current_node] if self.shortest_paths_in_cluster[node][T] == min_dis]
            next_node = random.choice(next_nodes)
            path.append(next_node)
        
        input_id = input_id + [encoder(node) for node in key_nodes] + [encoder(';')] + [encoder(node) for node in path] + [encoder("E")]
        if self.provide_planning:
            label = label + [-100] * len(key_nodes) + [-100] + [encoder(node) for node in path] + [encoder("E")]
        else:
            label = label + [encoder(node) for node in key_nodes] + [encoder(';')] + [encoder(node) for node in path] + [encoder("E")]
        attention_mask = attention_mask + [1]*len(key_nodes) + [1] + [1] * len(path) + [1]
        return input_id, label, attention_mask
    
    def get_train_data_TwolevelGraph_SFT4_planning3(self):
        (i,j)=random.choice(self.SFT1_train_pairs)
        return self.get_data_TwolevelGraph_SFT4_planning3((i, j))
    def get_eval_data_TwolevelGraph_SFT4_planning3(self):
        (i,j) = self.get_eval_pair_TwolevelGraph_SFT3()
        return self.get_data_TwolevelGraph_SFT4_planning3((i, j))

    def get_data_TwolevelGraph_SFT4(self, pair):    
        
        S, T = pair
        input_id = [encoder(S), encoder(T), encoder(":")]
        label = [-100] * 3
        attention_mask = [1] * 3

        cluster_S = (S-1) // self.N2
        cluster_T = (T-1) // self.N2

        upperlevel_path = self.upperlevel_shortest_path_seq[cluster_S][cluster_T]

        path = []
        current_node = S
        for i in range(len(upperlevel_path)-1):
            (M1, M2) = self.connecting_edge[(upperlevel_path[i],upperlevel_path[i+1])]
            path.append(current_node)
            while path[-1] != M1: # the shortest_path is only within cluster!!
                current_node = path[-1]
                min_dis = min(self.shortest_paths_in_cluster[n][M1] for n in self.adj_list_in_cluster[current_node])
                next_nodes = [node for node in self.adj_list_in_cluster[current_node] if self.shortest_paths_in_cluster[node][M1] == min_dis]
                next_node = random.choice(next_nodes)
                path.append(next_node)
            current_node = M2
        path.append(current_node)
        while path[-1] != T:
            current_node = path[-1]
            min_dis = min(self.shortest_paths_in_cluster[n][T] for n in self.adj_list_in_cluster[current_node])
            next_nodes = [node for node in self.adj_list_in_cluster[current_node] if self.shortest_paths_in_cluster[node][T] == min_dis]
            next_node = random.choice(next_nodes)
            path.append(next_node)
        
        input_id = input_id + [encoder(node) for node in path] + [encoder("E")]
        label = label + [encoder(node) for node in path] + [encoder("E")]
        attention_mask = attention_mask + [1] * len(path) + [1]
        return input_id, label, attention_mask
    def get_train_data_TwolevelGraph_SFT4(self):
        (i,j)=random.choice(self.SFT1_train_pairs)
        return self.get_data_TwolevelGraph_SFT4((i, j)) 
    def get_eval_data_TwolevelGraph_SFT4(self):
        (i,j) = self.get_eval_pair_TwolevelGraph_SFT3()
        return self.get_data_TwolevelGraph_SFT4((i, j))

    def get_data_TwolevelGraph_SFT3_planning2(self, pair):    
        
        S, T = pair
        input_id = [encoder(S), encoder(T), encoder(":")]
        label = [-100] * 3
        attention_mask = [1] * 3

        cluster_S = (S-1) // self.N2
        cluster_T = (T-1) // self.N2

        upperlevel_path = [cluster_S]
        while upperlevel_path[-1] != cluster_T:
            current_cluster = upperlevel_path[-1]
            min_dis = min(self.upperlevel_shortest_paths[N][cluster_T] for N in self.adj_list_across_cluster[current_cluster])
            next_clusters = [node for node in self.adj_list_across_cluster[current_cluster] if self.upperlevel_shortest_paths[node][cluster_T] == min_dis]
            next_cluster = random.choice(next_clusters)
            upperlevel_path.append(next_cluster)
        # print(upperlevel_path)

        path = []
        key_nodes = []
        current_node = S
        for i in range(len(upperlevel_path)-1):
            (M1, M2) = self.connecting_edge[(upperlevel_path[i],upperlevel_path[i+1])]
            path.append(current_node)
            while path[-1] != M1: # the shortest_path is only within cluster!!
                current_node = path[-1]
                min_dis = min(self.shortest_paths_in_cluster[n][M1] for n in self.adj_list_in_cluster[current_node])
                next_nodes = [node for node in self.adj_list_in_cluster[current_node] if self.shortest_paths_in_cluster[node][M1] == min_dis]
                next_node = random.choice(next_nodes)
                path.append(next_node)
            current_node = M2
            key_nodes.append(M1)
            key_nodes.append(M2)
        
        path.append(current_node)
        while path[-1] != T:
            current_node = path[-1]
            min_dis = min(self.shortest_paths_in_cluster[n][T] for n in self.adj_list_in_cluster[current_node])
            next_nodes = [node for node in self.adj_list_in_cluster[current_node] if self.shortest_paths_in_cluster[node][T] == min_dis]
            next_node = random.choice(next_nodes)
            path.append(next_node)
        
        if self.random_planning:
            if len(path) < len(key_nodes):
                key_nodes = path
                # breakpoint()
            else:
                indices = random.sample(range(len(path)), len(key_nodes))
                indices.sort()
                key_nodes = [path[i] for i in indices]
        if self.fix_interval:
            key_nodes = [path[i] for i in range(1, len(path), 2)]
        
        input_id = input_id + [encoder(node) for node in key_nodes] + [encoder(';')] 

        if self.task_type == "Two-levelGraph_SFT3_CEL":
            label = label + [-100] * len(key_nodes) + [-100]
            for i in range(min(self.stage, len(key_nodes) + 1)):
                label[-(i+1)] = input_id[-(i+1)]
            label = label + [encoder(node) for node in path] + [encoder("E")]
        elif self.provide_planning:
            label = label + [-100] * len(key_nodes) + [-100] + [encoder(node) for node in path] + [encoder("E")]
        else:
            label = label + [encoder(node) for node in key_nodes] + [encoder(';')] + [encoder(node) for node in path] + [encoder("E")]
        
        input_id = input_id + [encoder(node) for node in path] + [encoder("E")]
        
        attention_mask = attention_mask + [1]*len(key_nodes) + [1] + [1] * len(path) + [1]
        return input_id, label, attention_mask
    
    def get_train_data_TwolevelGraph_SFT3_planning2(self):
        (i,j)=random.choice(self.SFT1_train_pairs)
        return self.get_data_TwolevelGraph_SFT3_planning2((i, j))
    def get_eval_data_TwolevelGraph_SFT3_planning2(self):
        (i,j) = self.get_eval_pair_TwolevelGraph_SFT3()
        return self.get_data_TwolevelGraph_SFT3_planning2((i, j))

    def get_data_TwolevelGraph_SFT3(self, pair):    
        
        S, T = pair
        input_id = [encoder(S), encoder(T), encoder(":")]
        label = [-100] * 3
        attention_mask = [1] * 3

        cluster_S = (S-1) // self.N2
        cluster_T = (T-1) // self.N2

        upperlevel_path = [cluster_S]
        while upperlevel_path[-1] != cluster_T:
            current_cluster = upperlevel_path[-1]
            min_dis = min(self.upperlevel_shortest_paths[N][cluster_T] for N in self.adj_list_across_cluster[current_cluster])
            next_clusters = [node for node in self.adj_list_across_cluster[current_cluster] if self.upperlevel_shortest_paths[node][cluster_T] == min_dis]
            next_cluster = random.choice(next_clusters)
            upperlevel_path.append(next_cluster)

        path = []
        current_node = S
        for i in range(len(upperlevel_path)-1):
            (M1, M2) = self.connecting_edge[(upperlevel_path[i],upperlevel_path[i+1])]
            path.append(current_node)
            while path[-1] != M1: 
                current_node = path[-1]
                min_dis = min(self.shortest_paths_in_cluster[n][M1] for n in self.adj_list_in_cluster[current_node])
                next_nodes = [node for node in self.adj_list_in_cluster[current_node] if self.shortest_paths_in_cluster[node][M1] == min_dis]
                next_node = random.choice(next_nodes)
                path.append(next_node)
            current_node = M2
        path.append(current_node)
        while path[-1] != T:
            current_node = path[-1]
            min_dis = min(self.shortest_paths_in_cluster[n][T] for n in self.adj_list_in_cluster[current_node])
            next_nodes = [node for node in self.adj_list_in_cluster[current_node] if self.shortest_paths_in_cluster[node][T] == min_dis]
            next_node = random.choice(next_nodes)
            path.append(next_node)
        
        input_id = input_id + [encoder(node) for node in path] + [encoder("E")]
        label = label + [encoder(node) for node in path] + [encoder("E")]
        attention_mask = attention_mask + [1] * len(path) + [1]
        return input_id, label, attention_mask

    def get_eval_pair_TwolevelGraph_SFT3(self):
        if hasattr(self, "SFT1_eval_pairs"):
            (i,j)=random.choice(self.SFT1_eval_pairs)
        else:
            k1 = random.randint(0, self.N1-1)
            k2 = random.randint(0, self.N1-1)
            while k1 == k2:
                k2 = random.randint(0, self.N1-1)
            i = k1 * self.N2 + random.randint(1, self.N2)
            j = k2 * self.N2 + random.randint(1, self.N2)
            while ((i,j) in self.SFT1_train_pairs_set):
                k1 = random.randint(0, self.N1-1)
                k2 = random.randint(0, self.N1-1)
                while k1 == k2:
                    k2 = random.randint(0, self.N1-1)
                i = k1 * self.N2 + random.randint(1, self.N2)
                j = k2 * self.N2 + random.randint(1, self.N2)
        return (i, j)
    
    def get_train_data_TwolevelGraph_SFT3(self):
        (i,j)=random.choice(self.SFT1_train_pairs)
        return self.get_data_TwolevelGraph_SFT3((i, j)) 
    def get_eval_data_TwolevelGraph_SFT3(self):
        (i,j) = self.get_eval_pair_TwolevelGraph_SFT3()
        return self.get_data_TwolevelGraph_SFT3((i, j))

    def get_data_TwolevelGraph_SFT2_planning1(self, pair):
        
        S, T = pair
        input_id = [encoder(S), encoder(T), encoder(":")]
        label = [-100] * 3
        attention_mask = [1] * 3

        cluster_S = (S-1) // self.N2
        cluster_T = (T-1) // self.N2
        M1, M2 = self.connecting_edge[cluster_S][cluster_T]
        path = [S]
        while path[-1] != M1:
            current_node = path[-1]
            min_dis = min(self.shortest_paths_in_cluster[n][M1] for n in self.adj_list_in_cluster[current_node])
            next_nodes = [node for node in self.adj_list_in_cluster[current_node] if self.shortest_paths_in_cluster[node][M1] == min_dis]
            next_node = random.choice(next_nodes)
            path.append(next_node)
        path.append(M2)
        while path[-1] != T:
            current_node = path[-1]
            min_dis = min(self.shortest_paths_in_cluster[n][T] for n in self.adj_list_in_cluster[current_node])
            next_nodes = [node for node in self.adj_list_in_cluster[current_node] if self.shortest_paths_in_cluster[node][T] == min_dis]
            next_node = random.choice(next_nodes)
            path.append(next_node)

        input_id = input_id + [encoder(M1), encoder(M2)]+ [encoder(';')] + [encoder(node) for node in path] + [encoder("E")]
        if self.provide_planning == True:
            label = label + [-100] * 3 + [encoder(node) for node in path] + [encoder("E")]
        else:
            label = label + [encoder(M1), encoder(M2)]+ [encoder(';')]+ [encoder(node) for node in path] + [encoder("E")]
        attention_mask = attention_mask + [1] * 3 + [1] * len(path) + [1]
        return input_id, label, attention_mask

    def get_train_data_TwolevelGraph_SFT2_planning1(self):
        (i,j)=random.choice(self.SFT1_train_pairs)
        return self.get_data_TwolevelGraph_SFT2_planning1((i, j))

    def get_eval_data_TwolevelGraph_SFT2_planning1(self):
        (i,j)=random.choice(self.SFT1_eval_pairs)
        return self.get_data_TwolevelGraph_SFT2_planning1((i, j))

    def get_data_TwolevelGraph_SFT2(self, pair):
        
        S, T = pair
        input_id = [encoder(S), encoder(T), encoder(":")]
        label = [-100] * 3
        attention_mask = [1] * 3

        cluster_S = (S-1) // self.N2
        cluster_T = (T-1) // self.N2
        M1, M2 = self.connecting_edge[cluster_S][cluster_T]
        path = [S]
        while path[-1] != M1: # the shortest_path is only within cluster!!
            current_node = path[-1]
            min_dis = min(self.shortest_paths_in_cluster[n][M1] for n in self.adj_list_in_cluster[current_node])
            next_nodes = [node for node in self.adj_list_in_cluster[current_node] if self.shortest_paths_in_cluster[node][M1] == min_dis]
            next_node = random.choice(next_nodes)
            path.append(next_node)
        path.append(M2)
        while path[-1] != T:
            current_node = path[-1]
            min_dis = min(self.shortest_paths_in_cluster[n][T] for n in self.adj_list_in_cluster[current_node])
            next_nodes = [node for node in self.adj_list_in_cluster[current_node] if self.shortest_paths_in_cluster[node][T] == min_dis]
            next_node = random.choice(next_nodes)
            path.append(next_node)
        
        input_id = input_id + [encoder(node) for node in path] + [encoder("E")]
        label = label + [encoder(node) for node in path] + [encoder("E")]
        attention_mask = attention_mask + [1] * len(path) + [1]
        return input_id, label, attention_mask
    
    def get_train_data_TwolevelGraph_SFT2(self):
        (i,j)=random.choice(self.SFT1_train_pairs)
        return self.get_data_TwolevelGraph_SFT2((i, j))

    def get_eval_data_TwolevelGraph_SFT2(self):
        (i,j)=random.choice(self.SFT1_eval_pairs)
        return self.get_data_TwolevelGraph_SFT2((i, j))


    def get_data_TwolevelGraph_pt1(self):
        k = random.randint(0, self.N1-1)
        i = random.randint(1, self.N2)
        j = random.randint(1, self.N2)
        while i == j:
            j = random.randint(1, self.N2)
        
        S, T = k*self.N2+i, k*self.N2+j

        assert (S-1) // self.N2 == (T-1) // self.N2

        input_id = [encoder(S), encoder(T), encoder(":")]
        label = [-100] * 3
        attention_mask = [1] * 3
        
        path = [S]
        while path[-1] != T:
            current_node = path[-1]
            min_dis = min(self.shortest_paths_in_cluster[n][T] for n in self.adj_list_in_cluster[current_node])
            next_nodes = [node for node in self.adj_list_in_cluster[current_node] if self.shortest_paths_in_cluster[node][T] == min_dis]
            next_node = random.choice(next_nodes)
            path.append(next_node)

        input_id = input_id + [encoder(node) for node in path] + [encoder("E")]
        label = label + [encoder(node) for node in path] + [encoder("E")]
        attention_mask = attention_mask + [1] * len(path) + [1]
        return input_id, label, attention_mask

    
    def get_data_SFT1(self, pair):
        S, T = pair
        input_id = [encoder(S), encoder(T), encoder(":")]
        label = [-100] * 3
        attention_mask = [1] * 3
        
        path = [S]
        while path[-1] != T:
            current_node = path[-1]
            min_dis = min(self.shortest_paths[n][T] for n in self.adj_list[current_node])
            next_nodes = [node for node in self.adj_list[current_node] if self.shortest_paths[node][T] == min_dis]
            next_node = random.choice(next_nodes)
            path.append(next_node)
        input_id = input_id + [encoder(node) for node in path] + [encoder("E")]
        label = label + [encoder(node) for node in path] + [encoder("E")]
        attention_mask = attention_mask + [1] * len(path) + [1]
        return input_id, label, attention_mask
    

    def get_train_data_SFT3(self):
        pair = random.choice(self.SFT1_train_pairs)
        return self.get_data_SFT3(pair)
    
    def get_eval_data_SFT3(self):
        pair = random.choice(self.SFT1_eval_pairs)
        return self.get_data_SFT3(pair)

    def get_data_SFT3(self, pair):
        S, T = pair
        
        path = [S]
        while path[-1] != T:
            current_node = path[-1]
            min_dis = min(self.shortest_paths[n][T] for n in self.adj_list[current_node])
            next_nodes = [node for node in self.adj_list[current_node] if self.shortest_paths[node][T] == min_dis]
            next_node = random.choice(next_nodes)
            path.append(next_node)
        
        M = path[self.SFT_len//2]
        
        input_id = [encoder(S), encoder(T), encoder(":")] + [encoder(M)] + [encoder(":")] + [encoder(S), encoder(M), encoder(":")] + [encoder(node) for node in path[:self.SFT_len//2+1]] + [encoder(":")] + [encoder(M), encoder(T), encoder(":")] + [encoder(node) for node in path[self.SFT_len//2:]] + [encoder("E")]
        label = [-100]*3 + [encoder(M)] + [encoder(":")] + [encoder(S), encoder(M), encoder(":")] + [encoder(node) for node in path[:self.SFT_len//2+1]] + [encoder(":")] + [encoder(M), encoder(T), encoder(":")] + [encoder(node) for node in path[self.SFT_len//2:]] + [encoder("E")]
        attention_mask =[1] * len(input_id)
        assert len(input_id) == 3 + 2 + 3 + 3 + self.SFT_len + 1 + 1 + 1
        return input_id, label, attention_mask

    def get_data_pretrain1(self):
        while True:
            start_node = self.rng.randint(1, self.N)
            path = [start_node]
            for i in range(self.pretrain_len-1):
                # breakpoint()
                nodes = [node for node in self.adj_list[start_node] if node not in path]
                if len(nodes) == 0:
                    break
                next_node = self.rng.choice(nodes)
                path.append(next_node)
                start_node = next_node
            if len(path) == self.pretrain_len:
                break
        input_id = [encoder(node) for node in path]  # Here we do not have EOS for now.
        label = input_id
        attention_mask = [1] * len(input_id)
        return input_id, label, attention_mask

    def get_data_TwoTrees_pt1(self):
        idx = self.rng.randint(0, 1)
        i = self.rng.randint(0, self.N-1)
        j = self.rng.randint(0, self.N-1)
        while i == j:
            j = self.rng.randint(0, self.N-1)
        input_id = [encoder(self.Node_list[idx][i]),encoder(self.Node_list[idx][j]),encoder(":")]+[encoder(node) for node in self.paths[idx][i][j]] + [encoder("E")]
        label = [-100] * 3 + [encoder(node) for node in self.paths[idx][i][j]] + [encoder("E")]
        attention_mask = [1] * len(input_id)
        return input_id, label, attention_mask
    

    
    def get_train_data(self):
        if self.task_type == "pretrain1":
            return self.get_data_pretrain1()
        elif self.task_type == "SFT3":
            return self.get_train_data_SFT3()
        elif self.task_type == "TwoTrees_pt1":
            return self.get_data_TwoTrees_pt1()
        elif self.task_type in ["Two-levelGraph_pt1"]:
            return self.get_data_TwolevelGraph_pt1()
        elif self.task_type == "Two-levelGraph_SFT2":
            return self.get_train_data_TwolevelGraph_SFT2()
        elif self.task_type == "Two-levelGraph_SFT2_planning1":
            return self.get_train_data_TwolevelGraph_SFT2_planning1()
        elif self.task_type == "Two-levelGraph_SFT3":
            return self.get_train_data_TwolevelGraph_SFT3()
        elif self.task_type in ["Two-levelGraph_SFT3_planning2", "Two-levelGraph_SFT3_CEL"]:
            return self.get_train_data_TwolevelGraph_SFT3_planning2()
        elif self.task_type == "Two-levelGraph_SFT4":
            return self.get_train_data_TwolevelGraph_SFT4()
        elif self.task_type == "Two-levelGraph_SFT4_planning3":
            return self.get_train_data_TwolevelGraph_SFT4_planning3()
        elif self.task_type == "Two-levelGraph_SFT5":
            return self.get_train_data_TwolevelGraph_SFT5()
        elif self.task_type in ["Two-levelGraph_SFT5_planning4", "Two-levelGraph_onlyplanning4"] :
            return self.get_train_data_TwolevelGraph_SFT5_planning4()
        else:
            raise ValueError("Invalid task type")
    def get_eval_data(self):
        if self.task_type == "pretrain1":
            return self.get_data_pretrain1()
        elif self.task_type in ["SFT1", "SFT2", "SFT4", "SFT6", "SFT8"]:
            return self.get_eval_data_SFT1()
        elif self.task_type == "SFT3":
            return self.get_eval_data_SFT3()
        elif self.task_type == "SFT5":
            return self.get_eval_data_SFT5()
        elif self.task_type == "TwoTrees_pt1":
            return self.get_data_TwoTrees_pt1()
        elif self.task_type in ["Two-levelGraph_pt1"]:
            return self.get_data_TwolevelGraph_pt1()
        elif self.task_type == "Two-levelGraph_SFT2":
            return self.get_eval_data_TwolevelGraph_SFT2()
        elif self.task_type == "Two-levelGraph_SFT2_planning1":
            return self.get_eval_data_TwolevelGraph_SFT2_planning1()
        elif self.task_type == "Two-levelGraph_SFT3":
            return self.get_eval_data_TwolevelGraph_SFT3()
        elif self.task_type in ["Two-levelGraph_SFT3_planning2", "Two-levelGraph_SFT3_CEL"]:
            return self.get_eval_data_TwolevelGraph_SFT3_planning2()
        elif self.task_type == "Two-levelGraph_SFT4":
            return self.get_eval_data_TwolevelGraph_SFT4()
        elif self.task_type == "Two-levelGraph_SFT4_planning3":
            return self.get_eval_data_TwolevelGraph_SFT4_planning3()
        else:
            raise ValueError("Invalid task type")
    
    def check_valid(self, S, T, path): # Bett[er way is implement this within task class.
        # breakpoint() # print input_id, label, attention_mask
        if self.task_type in ["SFT1", "SFT2", "TwoTrees_SFT1", "TwoTrees_pt1", "TwoTrees_pt2", "TwoTrees_pt3", "TwoTrees_SFT2", "TwoTrees_SFT3", "Two-levelDumbbell_pt2", "Two-levelDumbbell_SFT1", "Two-levelDumbbell_SFT2", "Two-levelDumbbell_pt1", "Two-levelGraph_pt1", "Two-levelGraph_SFT1", "SFT4", "SFT5", "SFT6", "Two-levelGraph_SFT2", "SFT7", "SFT8", "Two-levelGraph_SFT3", "SFT9", "SFT10", "Two-levelGraph_SFT4", "Two-levelGraph_pt2"] or (self.task_type in ["Two-levelGraph_SFT2_planning1", "Two-levelGraph_SFT3_planning2", "Two-levelGraph_SFT4_planning3"] and self.provide_planning == True):
            if len(path) < 3: 
                return False
            if path[0] != S or path[-2] != T or path[-1]!= "E":
                return False
            for i in range(1, len(path) - 1):
                if (path[i-1],path[i]) not in self.edge_set:
                    return False
            return True
        elif self.task_type == "Two-levelGraph_SFT2_planning1" and self.provide_planning == False:
            if ';' in path:
                S_idx = path.index(';')
                path = path[S_idx+1:]
            else:
                path = path[2:]
            if len(path) < 3:
                return False
            if path[0] != S or path[-2] != T or path[-1]!= "E":
                return False
            for i in range(1, len(path) - 1):
                if (path[i-1],path[i]) not in self.edge_set:
                    return False
            return True
        elif (self.task_type in ["Two-levelGraph_SFT3_planning2","Two-levelGraph_SFT4_planning3"] and self.provide_planning == False) or self.task_type == "Two-levelGraph_SFT3_CEL":
            if ';' not in path and self.task_type != "Two-levelGraph_SFT3_CEL":
                return False
            if ';' in path:
                S_idx = path.index(';')
                path = path[S_idx+1:]
            if len(path) < 3:
                return False
            if path[0] != S or path[-2] != T or path[-1]!= "E":
                return False
            for i in range(1, len(path) - 1):
                if (path[i-1],path[i]) not in self.edge_set:
                    return False
            return True
        elif self.task_type == "Two-levelGraph_onlyplanning2":
            path = [node for node in path if node != '<pause>']
            if len(path) < 3:
                return False
            if (path[0]-1)//self.N2 != (S-1)//self.N2 or (path[-2]-1) // self.N2 != (T-1)//self.N2 or path[-1]!= "E":
                return False
            if (len(path) - 1) % 2 == 1:
                return False
            for i in range(0, len(path)-1, 2): # no E 
                if (path[i], path[i+1]) not in self.edge_set:
                    return False
            return True
        elif self.task_type == "Two-levelGraph_onlyplanning4":
            S_idx = (S-1) % self.N2 + 1
            T_idx = (T-1) % self.N2 + 1
            path = [node for node in path if node != '<pause>']
            if len(path) < 3:
                return False
            if path[0] != S or path[-2] != T or path[-1]!= "E":
                return False
            if (len(path) - 1) % 2 == 1:
                return False
            for i in range(0, len(path)-1, 2): # no E 
                if (path[i] - 1) % self.N2 + 1 != S_idx or (path[i+1] - 1) % self.N2 + 1 != T_idx:
                    return False
            for i in range(1, len(path)-2, 2):
                U_cluster = (path[i] - 1) // self.N2
                V_cluster = (path[i+1] - 1) // self.N2
                if self.upperlevel_shortest_paths[U_cluster][V_cluster] != 1:
                    return False
            return True        
        elif self.task_type == "SFT3":
            if len(path) != 2 + 3 + 3 + self.SFT_len + 1 + 1 + 1:
                return False
            if path[1] != ":" or path[2]!=S or path[4]!=':' or path[4+self.SFT_len//2+4]!=T or path[4+self.SFT_len//2+5]!=':' or path[-1]!='E':
                return False
            for i in range(5, 4+self.SFT_len//2+1):
                if (path[i], path[i+1]) not in self.edge_set:
                    return False
            for i in range(4+self.SFT_len//2+6, len(path) - 2):
                if (path[i], path[i+1]) not in self.edge_set:
                    return False
            return True

        else:
            raise ValueError("Invalid task type")

    def get_upperlevel_shortest_paths(self):
        self.upperlevel_shortest_paths = [[float('inf')] * (self.N1) for _ in range(self.N1)]
        for i in range(self.N1):
            self.upperlevel_shortest_paths[i][i] = 0
        for u, v in self.Outer_Edge_list:
            cluster_u = (u-1) // self.N2
            cluster_v = (v-1) // self.N2
            self.upperlevel_shortest_paths[cluster_u][cluster_v] = 1
            self.upperlevel_shortest_paths[cluster_v][cluster_u] = 1
        for k in range(self.N1):
            for i in range(self.N1):
                for j in range(self.N1):
                    if self.upperlevel_shortest_paths[i][j] > self.upperlevel_shortest_paths[i][k] + self.upperlevel_shortest_paths[k][j]:
                        self.upperlevel_shortest_paths[i][j] = self.upperlevel_shortest_paths[i][k] + self.upperlevel_shortest_paths[k][j]
        # breakpoint() # print upperlevel_shortest_paths
    
    def get_shortest_paths_in_cluster(self):
        self.adj_list_in_cluster = [[] for i in range(self.N + 1)]
        if self.graph_info["config"]["directed"]:
            for u, v in self.Edge_list:
                if (u-1) // self.N2 == (v-1) // self.N2:
                    self.adj_list_in_cluster[u].append(v)
        else: 
            for u, v in self.Edge_list:
                if (u-1) // self.N2 == (v-1) // self.N2:
                    self.adj_list_in_cluster[u].append(v)
                    self.adj_list_in_cluster[v].append(u)
        
        # self.shortest_paths_in_cluster = [[float('inf')] * (self.N + 1) for _ in range(self.N + 1)]
        self.shortest_paths_in_cluster = [{} for _ in range(self.N + 1)]
        for k in range(1, self.N+1, self.N2):
            for i in range(k, k+self.N2):
                for j in range(k, k+self.N2):
                    if i != j:
                        self.shortest_paths_in_cluster[i][j] = float('inf')
        
        for i in range(1, self.N + 1):
            self.shortest_paths_in_cluster[i][i] = 0
            queue = deque([i])
            while queue:
                node = queue.popleft()
                for neighbor in self.adj_list_in_cluster[node]:
                    if self.shortest_paths_in_cluster[i][neighbor] == float('inf'):
                        self.shortest_paths_in_cluster[i][neighbor] = self.shortest_paths_in_cluster[i][node] + 1
                        queue.append(neighbor)
        # breakpoint() # print self.shortest_paths_in_cluster
        
    def get_shortest_paths(self):
        self.shortest_paths = [[float('inf')] * (self.N + 1) for _ in range(self.N + 1)]
        for i in range(1, self.N + 1):
            self.shortest_paths[i][i] = 0
            queue = deque([i])
            while queue:
                node = queue.popleft()
                for neighbor in self.adj_list[node]:
                    if self.shortest_paths[i][neighbor] == float('inf'):
                        self.shortest_paths[i][neighbor] = self.shortest_paths[i][node] + 1
                        queue.append(neighbor)
    
    def update_stage(self):
        assert self.task_type == "Two-levelGraph_SFT3_CEL"
        self.stage += 1
    
    def prepare_data(self, S,T, path):
        input_id = [encoder(S), encoder(T), encoder(":")]
        label = [-100] * 3
        attention_mask = [1] * 3
        
        input_id = input_id + [encoder(node) for node in path] + [encoder("E")]
        label = label + [encoder(node) for node in path] + [encoder("E")]
        attention_mask = attention_mask + [1] * len(path) + [1]
        return input_id, label, attention_mask

    def prepare_data_planning(self, S,T, path, key_nodes):
        input_id = [encoder(S), encoder(T), encoder(":")]
        label = [-100] * 3
        attention_mask = [1] * 3
        
        input_id = input_id + [encoder(node) for node in key_nodes] + [encoder(';')] + [encoder(node) for node in path] + [encoder("E")]
        label = label + [encoder(node) for node in key_nodes] + [encoder(';')] + [encoder(node) for node in path] + [encoder("E")]
        attention_mask = attention_mask + [1]*len(key_nodes) + [1] + [1] * len(path) + [1]
        return input_id, label, attention_mask
    

    # def get_batch(self, batch_size: int):
    #     input_ids = []
    #     labels = []
    #     attention_masks = []
    #     for i in range(batch_size):
    #         input_id, label, attention_mask = self.get_strings()
    #         input_ids.append(input_id)
    #         labels.append(label)
    #         attention_masks.append(attention_mask)
        
    #     input_ids, attention_masks, labels = padding(input_ids, attention_masks=attention_masks, labels=labels, padding_direction="right")

    #     input_ids = torch.tensor(input_ids, dtype=torch.long)
    #     attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    #     labels = torch.tensor(labels, dtype=torch.long)

    #     breakpoint() # print input_ids, attention_masks, labels
    #     return input_ids, attention_masks, labels


