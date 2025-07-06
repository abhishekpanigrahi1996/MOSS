# Temporarily comment test_coverage
import sys
import os
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, parent_dir)
from ..task import GeneralPathFindingTask

import json
import random

class KPartiteGraphTaskGenerator(GeneralPathFindingTask):
    def get_node(self, k,i): # k \in [0, K-1], i \in [1, N]
        return i + k * self.N
    def get_layer_idx(self, node_id):
        return (node_id - 1) // self.N, (node_id - 1) % self.N + 1
    def dfs(self, u, w, num_layer, train_edge_set):
        if num_layer == self.K - 2:
            assert w in self.adj_list[u]
            train_edge_set.add((u, w))
            self.cnt += 1
            return
        for v in self.adj_list[u]:
            if w in self.num_paths[num_layer + 1][self.get_layer_idx(v)[1]]:
                train_edge_set.add((u, v))
                self.cnt += 1
                self.dfs(v, w, num_layer+1, train_edge_set)
    
    def test_coverage(self):
        self.cnt = 0
        train_edge_set = set()
        for (u,w) in self.train_pairs:
            self.dfs(u,w, 0, train_edge_set)
        print("number covered edges:", len(train_edge_set))
        print("number of edges train_pairs:", len(self.train_pairs))
        print("number of edges in the graph:", len(self.edge_set))
        print("numer of total training edges", self.cnt)
        return train_edge_set

    def dfs2(self,k,u):
        if k == 0:
            return u
        for v in self.reverse_adj_list[u]:
            w = self.dfs2(k-1, v)
            if w != None:
                return w
        return None

    def get_a_path(self,x,y): # find a path that go through x and y
        y_layer, y_idx = self.get_layer_idx(y)
        x_layer, x_idx = self.get_layer_idx(x) 
        if y_layer != self.K-1 and len(self.num_paths[y_layer][y_idx]) == 0:
            return None
        v = self.rng.choice(list(self.num_paths[y_layer][y_idx].keys())) if y_layer != self.K-1 else y # any node that v goes to.
        
        u = self.dfs2(x_layer,x)
        return (u,v) if u != None else None

    def __init__(self, config):
        random.seed(config.random_seed)
        self.config = config
        self.rng = random.Random(config.random_seed)
        # breakpoint()
        with open(config.graph_path, "r") as f:
            self.graph_info = json.load(f)
        self.task_type = config.task_type 
        assert self.task_type in ["KPartiteGraph1", "KPartiteGraph1_planning", "KPartiteGraph_pt1", "KPartiteGraph1_planning2"]

        if self.task_type in ["KPartiteGraph_pt1", "KPartiteGraph1_planning2"]:
            self.split_layer = config.split_layer
        if "planning" in self.task_type:
            self.planning_with_ST = config.planning_with_ST

        assert self.graph_info["config"]["Type"] in ["KPartiteGraph", "KPartiteGraph_Bernoulli"] 
        assert self.graph_info["config"]["directed"] == True

        ## Prepare the grpah.
        self.N = self.graph_info["config"]["N"]
        self.K = self.graph_info["config"]["K"]
        self.Edge_list = self.graph_info["Edge_list"]
        self.adj_list = [[] for i in range(self.K * self.N + 1)]
        self.reverse_adj_list = [[] for i in range(self.K * self.N + 1)]
        for u, v in self.Edge_list:
            self.adj_list[u].append(v)
            self.reverse_adj_list[v].append(u)
        self.edge_set = set()
        for u, v in self.Edge_list:
            self.edge_set.add((u, v))
        
        ## Prepare the task.
        if self.task_type in ["KPartiteGraph1", "KPartiteGraph1_planning", "KPartiteGraph1_planning2"]:
            self.eval_rate = config.eval_rate
            self.prepare_SFT1()
        else:
            self.prepare_PT1()

    def get_pairwise_num_path(self):
        num_paths = [[[{} for i in range(self.N+1)] for k2 in range(k1+1)] for k1 in range(self.K)]
        for k1 in range(self.K):
            for i in range(1, self.N+1):
                num_paths[k1][k1][i][self.get_node(k1,i)] = 1
        for k1 in range(self.K):
            for k2 in range(k1-1, -1, -1):
                for i in range(1, self.N+1):
                    u = self.get_node(k2, i)
                    for v in self.adj_list[u]:
                        _, j = self.get_layer_idx(v)
                        assert _ == k2+1
                        for (w, num_paths_vw) in num_paths[k1][k2+1][j].items():
                            if w not in num_paths[k1][k2][i]:
                                num_paths[k1][k2][i][w] = 0
                            num_paths[k1][k2][i][w] += num_paths_vw
        return num_paths

    def prepare_PT1(self):
        self.num_paths = self.get_pairwise_num_path()
        ## prepare the graph.
        # for k1 in range(self.K):
        #     for k2 in range(k1+1):
        #         for i in range(1, self.N+1):
        #             print("k1", k1, "k2", k2, "i", i)
        #             print(self.num_paths[k1][k2][i])
        # breakpoint()

        self.train_pairs = []
        assert len(self.split_layer) == 1
        self.split_layer = self.split_layer[0]
        # breakpoint()
        for i in range(1, self.N+1):
            for (w, _) in self.num_paths[self.split_layer][0][i].items():
                self.train_pairs.append((i, w))
        for i in range(1, self.N+1):
            for (w, _) in self.num_paths[self.K-1][self.split_layer][i].items():
                self.train_pairs.append((self.get_node(self.split_layer,i), w))
        # print(self.train_pairs)
        # breakpoint()

    def prepare_SFT1(self):
        ## prepare the graph.
        self.num_paths = [[{} for i in range(self.N+1)] for k in range(self.K-1)] # (k,i,v) number of paths from (k,i) to v. Here v is the true node id, not the k-partite node id.
        for i in range(1, self.N+1):
            u = self.get_node(self.K-2, i)
            for v in self.adj_list[u]:
                if v not in self.num_paths[self.K-2][i]:
                    self.num_paths[self.K-2][i][v] = 0
                self.num_paths[self.K-2][i][v] = 1

        for k in range(self.K-1 -2, -1, -1):
            for i in range(1, self.N+1):
                u = self.get_node(k, i)
                for v in self.adj_list[u]:
                    _, j = self.get_layer_idx(v)
                    assert _ == k+1
                    for (w, num_paths_vw) in self.num_paths[k+1][j].items():
                        if w not in self.num_paths[k][i]:
                            self.num_paths[k][i][w] = 0
                        self.num_paths[k][i][w] += num_paths_vw
        self.pairwise_num_paths = self.get_pairwise_num_path()

        self.train_pairs = []
        self.eval_pairs = []
        for i in range(1, self.N+1):
            for (w, _) in self.num_paths[0][i].items():
                if self.rng.random() < 1 - self.eval_rate:
                    self.train_pairs.append((i, w))
                else:
                    self.eval_pairs.append((i, w))
        # train_edge_set = self.test_coverage() # Temporarily
        # for (u, v) in self.Edge_list: 
        #     if (u, v) not in train_edge_set:
        #         pair = self.get_a_path(u,v)
        #         if pair is not None:
        #             self.train_pairs.append(pair)
        # self.test_coverage()
        # breakpoint()

    def get_data(self,S,T):
        path = [S]
        x = S
        while len(path) < self.K-1:
            node_list = []
            num_paths = []
            for node in self.adj_list[x]:
                if T in self.num_paths[self.get_layer_idx(node)[0]][self.get_layer_idx(node)[1]]:
                    node_list.append(node)
                    num_paths.append(self.num_paths[self.get_layer_idx(node)[0]][self.get_layer_idx(node)[1]][T])
            node = self.rng.choices(node_list, weights=num_paths, k=1)[0]
            path.append(node)
            x = node
        path.append(T)
        if self.task_type == "KPartiteGraph1_planning":
            planning_idx = len(path)//2
            planning = [path[planning_idx-1], path[planning_idx]]
        elif self.task_type == "KPartiteGraph1_planning2":
            planning = [path[split_layer] for split_layer in self.split_layer]
        if "planning" in self.task_type and self.planning_with_ST == True:
            planning = [S] + planning + [T]
        if self.task_type == "KPartiteGraph1":
            return self.prepare_data(S, T, path)
        elif self.task_type in ["KPartiteGraph1_planning", "KPartiteGraph1_planning2"]:
            return self.prepare_data_planning(S, T, path, planning)
        
    def get_data_pt1(self, S, T):
        S_layer, S_idx = self.get_layer_idx(S)
        T_layer, T_idx = self.get_layer_idx(T)
        path = [S]
        x = S

        while len(path) < T_layer - S_layer:
            node_list = []
            num_paths = []
            for node in self.adj_list[x]:
                if T in self.num_paths[T_layer][S_layer + len(path)][self.get_layer_idx(node)[1]]:
                    node_list.append(node)
                    num_paths.append(self.num_paths[T_layer][S_layer + len(path)][self.get_layer_idx(node)[1]][T])
            node = self.rng.choices(node_list, weights=num_paths, k=1)[0]
            path.append(node)
            x = node
        path.append(T)
        print(self.prepare_data(S, T, path))
        # breakpoint()
        return self.prepare_data(S, T, path)
    def get_train_data(self):
        if self.task_type == "KPartiteGraph_pt1":
            u,v = self.rng.choice(self.train_pairs)
            return self.get_data_pt1(u,v)
        elif self.task_type in ["KPartiteGraph1", "KPartiteGraph1_planning", "KPartiteGraph1_planning2"]:
            u,v = self.rng.choice(self.train_pairs)
            return self.get_data(u,v)
        u, v = self.rng.choice(self.train_pairs)
        return self.get_data(u,v)
    def get_eval_data(self):
        u,v = self.rng.choice(self.eval_pairs)
        return self.get_data(u,v)

    def check_valid(self,S,T,path):
        if self.task_type in ["KPartiteGraph1_planning", "KPartiteGraph1_planning2"]:
            if ';' not in path:
                return False
            S_idx = path.index(';')
            path = path[S_idx+1:]
            for i in range(len(path)-1):
                if not isinstance(path[i], int):
                    print("type noneint")
                    return False

        if len(path) < 3: 
            return False
        if path[0] != S or path[-2] != T or path[-1]!= "E":
            return False
        for i in range(1, len(path) - 1):
            if (path[i-1],path[i]) not in self.edge_set:
                return False
        return True