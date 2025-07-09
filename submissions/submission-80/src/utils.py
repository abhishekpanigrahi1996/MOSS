import argparse
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--graph_dir', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--graph_data_dir', type=str, default=None)

    parser.add_argument('--train_type', type=str, default='None',
                        choices = ["pretrain1", "pretrain2", "SFT1", "SFT2", "SFT3",
                                   "TwoTrees_pt1", "TwoTrees_pt2", "TwoTrees_pt3", "TwoTrees_SFT1", "TwoTrees_SFT2",
                                   "TwoTrees_SFT3", "Two-levelDumbbell_pt1", "Two-levelDumbbell_pt2",
                                   "Two-levelDumbbell_SFT1", "Two-levelDumbbell_SFT2", "Two-levelGraph_pt1", "Two-levelGraph_SFT1",
                                   "SFT4", "SFT5", "SFT6", "Two-levelGraph_SFT2", "Two-levelGraph_SFT2_planning1",
                                   "SFT7", "SFT8", "Two-levelGraph_SFT3", "Two-levelGraph_SFT3_planning2", "SFT9",
                                   "Two-levelGraph_onlyplanning2", "SFT10", "Two-levelGraph_SFT4", "Two-levelGraph_SFT4_planning3",
                                   "Two-levelGraph_SFT5", "Two-levelGraph_SFT5_planning4", "Two-levelGraph_onlyplanning4",
                                   "ReplicatedGraph_SFT1", "ReplicatedGraph_SFT1_planning", "ReplicatedGraph_SFT2", "ReplicatedGraph_SFT2_planning", "Two-levelGraph_SFT3_CEL",
                                   "KPartiteGraph1", "KPartiteGraph1_planning", "KPartiteGraph_pt1", "KPartiteGraph1_planning2", "Two-levelGraph_pt2"])
    parser.add_argument('--provide_planning', action='store_true')
    parser.add_argument('--pause_token', action='store_true')
    parser.add_argument('--random_planning', action='store_true')
    parser.add_argument('--fix_interval', action='store_true')
    parser.add_argument('--planning_with_cluster_token', action='store_true')
    parser.add_argument('--planning_with_ST', action='store_true')
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--N1', type=int, default=None)
    parser.add_argument('--N2', type=int, default=None)
    parser.add_argument('--graph_type1', type=str, default=None, choices=["Tree"])
    parser.add_argument('--graph_type2', type=str, default=None, choices=["Tree","Tree_with_additional_edge"])
    parser.add_argument('--edge_probability', type=float, default=None)
    parser.add_argument('--pretrain_len', type=int, default=None)
    parser.add_argument('--SFT_len', type=int, default=None)
    parser.add_argument('--fixed_len', type=int, default=None)
    parser.add_argument('--eval_rate', type=float, default=None)
    parser.add_argument('--num_train_path', type=int, default=None)
    parser.add_argument('--mix_probability', type=float, default=None)
    parser.add_argument('--only_detour_to_trainpairs', action='store_true')
    parser.add_argument('--graph_type', type=str, default='Bernoulli', choices = ["Bernoulli", "Tree", "TwoTrees", "Two-levelDumbbell", "Two-levelGraph", "Directed_Bernoulli", "ReplicatedGraph", "KPartiteGraph", "KPartiteGraph_Bernoulli"])
    parser.add_argument('--train_pair_path', type=str, default=None)
    parser.add_argument('--mix_pt_rate', type=float, default=0)
    parser.add_argument('--split_layer', type=int, nargs='+', default=None)


    parser.add_argument('--eval_pair_path', type=str, default=None)
    parser.add_argument('--max_generation_length', type=int, default=300)

    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--log_steps', type=int, default=32)
    parser.add_argument('--save_steps', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--report_to_wandb', action='store_true')
    parser.add_argument('--pid', type=int, default=0)
    parser.add_argument('--world_size', type = int, default = 1)   
    parser.add_argument('--save_dir', type=str, default = None)
    parser.add_argument('--eval_steps', type = int, default = 32)
    parser.add_argument('--eval_size', type = int, default = 512)
    
    parser.add_argument('--model_config_path', type=str, default=None)
    parser.add_argument('--num_hidden_layers', type = int, default = 1)
    parser.add_argument('--num_attention_heads', type = int, default = 1)
    parser.add_argument('--hidden_size', type = int, default=768)
    parser.add_argument('--vocab_size', type=int, default=200)
    parser.add_argument('--onehot_embed', action='store_true')
    parser.add_argument('--position_embedding', type=str, default='learned', choices=['learned', 'NoPE'])
    parser.add_argument('--extra_message', type=str, default=None)
    return parser.parse_args()

def encoder(t): # zero is used to padding.
    if t == ',':
        return 1
    elif t == ':':
        return 2
    elif t == ';':
        return 3
    elif t == 'B': # backtrack
        return 5
    elif t == 'E': # EOS
        return 6
    elif t == "Type0":
        return 7
    elif t == "Type1":
        return 8
    elif t == '<pause>':
        return 9
    else:
        return t + 10 # must be an integer.
    
def decoder(t): # zero is used to padding.
    if t == 0:
        return '<PAD>'
    elif t == 1:
        return ','
    elif t == 2:
        return ':'
    elif t == 3:
        return ';'
    elif t == 5:
        return 'B'
    elif t == 6:
        return 'E'
    elif t == 7:
        return "Type0"
    elif t == 8:
        return "Type1"
    elif t == 9:
        return '<pause>'
    else:
        return t - 10 # must be an integer.
    
