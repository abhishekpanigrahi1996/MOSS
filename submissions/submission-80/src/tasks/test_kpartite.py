from kpartite_graph_task_generator import KPartiteGraphTaskGenerator
from munch import Munch
config=Munch({
    "random_seed": 0,
    "graph_path": "data/Exp2_GeneralPathFinding/Graphs/KPartiteGraph_4_5_2.json",
    "task_type": "KPartiteGraph_pt1",
    "split_layer": 1,
})
task_generator = KPartiteGraphTaskGenerator(config)
for i in range(10):
    task_generator.get_train_data()
for i in range(10):
    task_generator.get_eval_data()