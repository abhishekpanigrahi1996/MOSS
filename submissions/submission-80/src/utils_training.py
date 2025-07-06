
from munch import Munch
from .task import GeneralPathFindingTask
from .tasks.kpartite_graph_task_generator import KPartiteGraphTaskGenerator
from torch.utils.data import IterableDataset
import torch
import random

class CustomDataCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_length = max(len(f['input_ids']) for f in features)
        
        for f in features:
            padding_length = max_length - len(f['input_ids'])

            f['attention_mask'] = f.get('attention_mask', [1] * len(f['input_ids'])) + [0] * padding_length

            f['input_ids'] = f['input_ids'] + [self.pad_token_id] * padding_length

            # If 'labels' are provided, pad them as well
            if 'labels' in f:
                f['labels'] = f['labels'] + [self.pad_token_id] * padding_length

        input_ids = torch.tensor([f['input_ids'] for f in features])
        attention_mask = torch.tensor([f['attention_mask'] for f in features])

        # breakpoint()
        if 'labels' in features[0]:
            labels = torch.tensor([f['labels'] for f in features])
            # breakpoit()
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        else:
            return {"input_ids": input_ids, "attention_mask": attention_mask}

def padding(input_ids, pad_token_id=0, padding_direction="left"):
    max_length = max(len(seq) for seq in input_ids)
    padded_input_ids = []
    attention_mask = []
    for seq in input_ids:
        padding_length = max_length - len(seq)
        if padding_direction == "right":
            padded_seq = seq + [pad_token_id] * padding_length
            mask = [1] * len(seq) + [0] * padding_length
        elif padding_direction == "left":
            padded_seq = [pad_token_id] * padding_length + seq
            mask = [0] * padding_length + [1] * len(seq)
        else:
            raise ValueError("padding_direction should be 'left' or 'right'")
        padded_input_ids.append(padded_seq)
        attention_mask.append(mask)
    return padded_input_ids, attention_mask


class CustomDataset(IterableDataset):
    def __init__(self, generator: GeneralPathFindingTask, type: str):
        self.generator = generator
        self.type = type

    def __iter__(self):
        while True:
            if self.type == "train":
                input_id, label, attention_mask = self.generator.get_train_data()
            elif self.type == "eval":
                input_id, label, attention_mask = self.generator.get_eval_data()
            else:
                raise ValueError("Invalid type")
            yield {"input_ids": input_id, "labels": label, "attention_mask": attention_mask}


def get_task_config(args, graph_path):
    if args.train_type == "Two-levelGraph_pt1":
        config=Munch({
            "random_seed": args.random_seed,
            "graph_path": graph_path,
            "task_type": args.train_type,
        })
    elif args.train_type in ["Two-levelGraph_SFT1", "Two-levelGraph_SFT2", "Two-levelGraph_SFT2_planning1", "Two-levelGraph_SFT3", "Two-levelGraph_SFT3_planning2", "Two-levelGraph_onlyplanning2", "Two-levelGraph_SFT4", "Two-levelGraph_SFT4_planning3", "Two-levelGraph_SFT5", "Two-levelGraph_SFT5_planning4", "Two-levelGraph_onlyplanning4", "Two-levelGraph_SFT3_CEL"]:
        config=Munch({
            "random_seed": args.random_seed,
            "graph_path": graph_path,
            "task_type": args.train_type,
            "eval_rate": args.eval_rate,
        })
        if args.train_type in ["Two-levelGraph_SFT2_planning1", "Two-levelGraph_SFT3_planning2", "Two-levelGraph_SFT4_planning3", "Two-levelGraph_SFT5_planning4", ]:
            config.update({"provide_planning": args.provide_planning})
        if args.train_type in ["Two-levelGraph_SFT5_planning4", "Two-levelGraph_SFT3_planning2"]: # currently only implemented from SFT5_planning4
            config.update({"random_planning": args.random_planning})
            config.update({"planning_with_cluster_token": args.planning_with_cluster_token})
            config.update({"fix_interval": args.fix_interval})  
        if args.train_type in ["Two-levelGraph_onlyplanning2", "Two-levelGraph_onlyplanning4"]:
            config.update({"pause_token": args.pause_token})
    elif args.train_type in ["KPartiteGraph1","KPartiteGraph1_planning", "KPartiteGraph1_planning2"]:
        config=Munch({
            "random_seed": args.random_seed,
            "graph_path": graph_path,
            "task_type": args.train_type,
            "eval_rate": args.eval_rate,
        })
        if args.train_type == "KPartiteGraph1_planning2":
            config.update({"split_layer": args.split_layer})
        if "planning" in args.train_type:
            config.update({"planning_with_ST": args.planning_with_ST})
    
    elif args.train_type in ["KPartiteGraph_pt1"]:
        config=Munch({
            "random_seed": args.random_seed,
            "graph_path": graph_path,
            "task_type": args.train_type,
            "split_layer": args.split_layer,
        })
    elif args.train_type == "Two-levelGraph_pt2":
        config=Munch({
            "random_seed": args.random_seed,
            "graph_path": graph_path,
            "task_type": args.train_type,
        })
    if args.random_planning == True or args.planning_with_cluster_token == True:
        assert args.train_type in ["Two-levelGraph_SFT5_planning4", "Two-levelGraph_SFT3_planning2"]
        assert not (args.random_planning and args.planning_with_cluster_token)
    return config

def get_graph_path(args):
    if args.graph_data_dir == None:
        if args.graph_type == "Bernoulli":
            graph_path = f"Bernoulli_{args.N}_{args.edge_probability}"
        elif args.graph_type == "Directed_Bernoulli":
            graph_path = f"Directed_Bernoulli_{args.N}_{args.edge_probability}"
        elif args.graph_type == "Tree":
            graph_path = f"Tree_{args.N}"
        elif args.graph_type == "TwoTrees":
            graph_path = f"TwoTrees_{args.N}"
        elif args.graph_type == "Two-levelDumbbell":
            graph_path = f"Two-levelDumbbell_{args.N1}_{args.N2}"
        elif args.graph_type == "Two-levelGraph":
            graph_path = f"Two-levelGraph_{args.graph_type1}_{args.graph_type2}_{args.N1}_{args.N2}"
        else:
            raise ValueError("Invalid graph_type")
    else:
        graph_path = args.graph_data_dir.split('/')[-1][:-5]  # remove .json
    return graph_path

def get_output_dir(args, graph_path, base_model_name=None):
    if args.train_type == "Two-levelGraph_pt1":
        output_dir = f"{args.output_dir}/tlg_pt1_{graph_path}@{args.max_steps}_LR={args.lr}_WD={args.weight_decay}_WR{args.warmup_ratio}_BS{args.batch_size}_#L={args.num_hidden_layers}_#H={args.num_attention_heads}_PE{args.position_embedding}_HS{args.hidden_size}"
    elif args.train_type == "Two-levelGraph_SFT1":
        output_dir = f"{args.output_dir}/tlg_sft1_{'PT' if args.model_dir==None else 'FT'}_{graph_path}@{args.eval_rate}_{args.max_steps}_LR={args.lr}_WD={args.weight_decay}_WR{args.warmup_ratio}_BS{args.batch_size}_#L={args.num_hidden_layers}_#H={args.num_attention_heads}_PE{args.position_embedding}_HS{args.hidden_size}{'_from'+base_model_name if args.model_dir != None else ''}"
    elif args.train_type == "Two-levelGraph_SFT2":
        output_dir = f"{args.output_dir}/tlg_sft2_{'PT' if args.model_dir==None else 'FT'}_{graph_path}@{args.eval_rate}_{args.max_steps}_LR={args.lr}_WD={args.weight_decay}_WR{args.warmup_ratio}_BS{args.batch_size}_#L={args.num_hidden_layers}_#H={args.num_attention_heads}_PE{args.position_embedding}_HS{args.hidden_size}{'_from'+base_model_name if args.model_dir != None else ''}"  
    elif args.train_type == "Two-levelGraph_SFT2_planning1":
        output_dir = f"{args.output_dir}/tlg_sft2planning1{'_provided' if args.provide_planning else ''}_{'PT' if args.model_dir==None else 'FT'}_{graph_path}@{args.eval_rate}_{args.max_steps}_LR={args.lr}_WD={args.weight_decay}_WR{args.warmup_ratio}_BS{args.batch_size}_#L={args.num_hidden_layers}_#H={args.num_attention_heads}_PE{args.position_embedding}_HS{args.hidden_size}{'_from'+base_model_name if args.model_dir != None else ''}"
    elif args.train_type == "Two-levelGraph_SFT3":
        output_dir = f"{args.output_dir}/tlg_sft3_{'PT' if args.model_dir==None else 'FT'}_{graph_path}@{args.eval_rate}_{args.max_steps}_LR={args.lr}_WD={args.weight_decay}_WR{args.warmup_ratio}_BS{args.batch_size}_#L={args.num_hidden_layers}_#H={args.num_attention_heads}_PE{args.position_embedding}_HS{args.hidden_size}{'_from'+base_model_name if args.model_dir != None else ''}"  
    elif args.train_type == "Two-levelGraph_SFT3_planning2":
        output_dir = f"{args.output_dir}/tlg_sft3planning2{'_random' if args.random_planning else ''}{'_fix_interval' if args.fix_interval else ''}{'_provided' if args.provide_planning else ''}_{'PT' if args.model_dir==None else 'FT'}_{graph_path}@{args.eval_rate}_{args.max_steps}_LR={args.lr}_WD={args.weight_decay}_WR{args.warmup_ratio}_BS{args.batch_size}_#L={args.num_hidden_layers}_#H={args.num_attention_heads}_PE{args.position_embedding}_HS{args.hidden_size}{'_from'+base_model_name if args.model_dir != None else ''}"
    elif args.train_type == "Two-levelGraph_SFT4_planning3":
        output_dir = f"{args.output_dir}/tlg_sft4planning3_{'PT' if args.model_dir==None else 'FT'}_{graph_path}@{args.eval_rate}_{args.max_steps}_LR={args.lr}_WD={args.weight_decay}_WR{args.warmup_ratio}_BS{args.batch_size}_#L={args.num_hidden_layers}_#H={args.num_attention_heads}_PE{args.position_embedding}_HS{args.hidden_size}{'_from'+base_model_name if args.model_dir != None else ''}"  
    elif args.train_type == "Two-levelGraph_onlyplanning2":
        output_dir = f"{args.output_dir}/tlg_onlyplanning2_{'PT' if args.model_dir==None else 'FT'}_{graph_path}@{args.eval_rate}_{args.max_steps}_LR={args.lr}_WD={args.weight_decay}_WR{args.warmup_ratio}_BS{args.batch_size}_#L={args.num_hidden_layers}_#H={args.num_attention_heads}_PE{args.position_embedding}_HS{args.hidden_size}{'_from'+base_model_name if args.model_dir != None else ''}" 
    elif args.train_type == "Two-levelGraph_SFT4":
        output_dir = f"{args.output_dir}/tlg_sft4_{'PT' if args.model_dir==None else 'FT'}_{graph_path}@{args.eval_rate}_{args.max_steps}_LR={args.lr}_WD={args.weight_decay}_WR{args.warmup_ratio}_BS{args.batch_size}_#L={args.num_hidden_layers}_#H={args.num_attention_heads}_PE{args.position_embedding}_HS{args.hidden_size}{'_from'+base_model_name if args.model_dir != None else ''}" 
    elif args.train_type == "Two-levelGraph_SFT5":
        output_dir = f"{args.output_dir}/tlg_sft5_{'PT' if args.model_dir==None else 'FT'}_{graph_path}@{args.eval_rate}_{args.max_steps}_LR={args.lr}_WD={args.weight_decay}_WR{args.warmup_ratio}_BS{args.batch_size}_#L={args.num_hidden_layers}_#H={args.num_attention_heads}_PE{args.position_embedding}_HS{args.hidden_size}{'_from'+base_model_name if args.model_dir != None else ''}" 
    elif args.train_type == "Two-levelGraph_SFT5_planning4":
        output_dir = f"{args.output_dir}/tlg_sft5planning4{'_random' if args.random_planning else ''}{'_fix_interval' if args.fix_interval else ''}{'cluster_token' if args.planning_with_cluster_token else ''}{'_provided' if args.provide_planning else ''}_{'PT' if args.model_dir==None else 'FT'}_{graph_path}@{args.eval_rate}_{args.max_steps}_LR={args.lr}_WD={args.weight_decay}_WR{args.warmup_ratio}_BS{args.batch_size}_#L={args.num_hidden_layers}_#H={args.num_attention_heads}_PE{args.position_embedding}_HS{args.hidden_size}{'_from'+base_model_name if args.model_dir != None else ''}"  
    elif args.train_type == "KPartiteGraph1":
        output_dir = f"{args.output_dir}/kpg1_{'PT' if args.model_dir==None else 'FT'}_{graph_path}@{args.eval_rate}_{args.max_steps}_LR={args.lr}_WD={args.weight_decay}_WR{args.warmup_ratio}_BS{args.batch_size}_#L={args.num_hidden_layers}_#H={args.num_attention_heads}_PE{args.position_embedding}_HS{args.hidden_size}{'_from'+base_model_name if args.model_dir != None else ''}"
    elif args.train_type == "KPartiteGraph1_planning":
        output_dir = f"{args.output_dir}/kpg1_planning{'_pwST' if args.planning_with_ST else ''}_{'PT' if args.model_dir==None else 'FT'}_{graph_path}@{args.eval_rate}_{args.max_steps}_LR={args.lr}_WD={args.weight_decay}_WR{args.warmup_ratio}_BS{args.batch_size}_#L={args.num_hidden_layers}_#H={args.num_attention_heads}_PE{args.position_embedding}_HS{args.hidden_size}{'_from'+base_model_name if args.model_dir != None else ''}"
    elif args.train_type == "KPartiteGraph_pt1":
        output_dir = f"{args.output_dir}/kpg_pt1_{'PT' if args.model_dir==None else 'FT'}_{graph_path}@{args.split_layer}_{args.max_steps}_LR={args.lr}_WD={args.weight_decay}_WR{args.warmup_ratio}_BS{args.batch_size}_#L={args.num_hidden_layers}_#H={args.num_attention_heads}_PE{args.position_embedding}_HS{args.hidden_size}{'_from'+base_model_name if args.model_dir != None else ''}"
        
    elif args.train_type == "KPartiteGraph1_planning2":
        output_dir = f"{args.output_dir}/kpg1_planning2{'_pwST' if args.planning_with_ST else ''}_{'PT' if args.model_dir==None else 'FT'}_{graph_path}@{args.eval_rate}_{args.split_layer}_{args.max_steps}_LR={args.lr}_WD={args.weight_decay}_WR{args.warmup_ratio}_BS{args.batch_size}_#L={args.num_hidden_layers}_#H={args.num_attention_heads}_PE{args.position_embedding}_HS{args.hidden_size}{'_from'+base_model_name if args.model_dir != None else ''}"
    elif args.train_type == "Two-levelGraph_pt2":
        output_dir = f"{args.output_dir}/tlg_pt2_{graph_path}@{args.max_steps}_LR={args.lr}_WD={args.weight_decay}_WR{args.warmup_ratio}_BS{args.batch_size}_#L={args.num_hidden_layers}_#H={args.num_attention_heads}_PE{args.position_embedding}_HS{args.hidden_size}"
    else:
        raise ValueError("Invalid train_type")
    
    insert_idx = output_dir.find("_LR")
    if insert_idx != -1: # Include random seed.
        output_dir = output_dir[:insert_idx] + f"_rs{args.random_seed}" + output_dir[insert_idx:]
    return output_dir

def get_task_generator(args, graph_path):
    if args.train_type in ["KPartiteGraph1", "KPartiteGraph1_planning","KPartiteGraph_pt1", "KPartiteGraph1_planning2"]:
        return KPartiteGraphTaskGenerator(get_task_config(args, graph_path))
    return GeneralPathFindingTask(get_task_config(args, graph_path))