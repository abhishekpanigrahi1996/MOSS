import torch
import numpy as np
import argparse


import torch
import numpy as np
import argparse

import logging
import os


def str_to_bool(x):
    if isinstance(x, bool):
        return x
    if x.lower() in ['true', '1', 'yes']:
        return True
    elif x.lower() in ['false', '0', 'no']:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false)")


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # markov chain and data params
    parser.add_argument('--order', default=1, type=int)
    parser.add_argument('--V', default=203, type=int)             #### fix the default value
    parser.add_argument('--M', default=100, type=int)             #### fix the default value        
    parser.add_argument('--seq_len', default=50, type=int)             #### fix the default value
    parser.add_argument('--initial', default='stationary', choices=['stationary', 'uniform', 'custom'])
    parser.add_argument('--num_chain_tmpl', default=1, type=int)             #### fix the default value
    parser.add_argument('--num_pos_tmpl', default=10, type=int)
    parser.add_argument('--bi_pos', default="random", type=str, choices=['random', 'fixed'], help="for target token to always appear at a fixed position at the end of the seq, pass \"fixed\"")
    parser.add_argument('--chain_per_pos', default=1, type=int)
    parser.add_argument('--ood_frac', default=0.5, type=float)
    parser.add_argument('--insertion_mode', default="shift", type=str, choices=['shift', 'replace'], help="should knowledge tokens replace the markov tokens or shift them....")

    parser.add_argument('--test_size', default=256, type=int)
    parser.add_argument('--ood_struct_eval', type=str_to_bool, default=False)             #### fix the default value
    
    parser.add_argument("--skip_spectok", action="store_true", help="don't add EOS, BOS, ...")

    # model params
    # parser.add_argument('--rpe', default="true", type=str, help='whether to use relative PE or not', choices=['false', 'true'])
    parser.add_argument('--rpe', type=str_to_bool, default=True, help='whether to use relative PE or not')

    parser.add_argument('--model', default='base', choices=['base'])
    parser.add_argument('--use_pretrained', default="none", type=str) # 'none', 'gpt-2' or a path to the pretrained model
    parser.add_argument('--dropout', default=0.1, type=float) #0.2
    parser.add_argument('--n_head', default=2, type=int)
    parser.add_argument('--n_layer', default=2, type=int) # depths in att + ff blocks
    parser.add_argument('--n_embd', default=32, type=int) # embedding size / hidden size ... 
    parser.add_argument('--context_length', default=256, type=int)
    parser.add_argument('--dtype', default=torch.float16, type=torch.dtype) #changed!
    parser.add_argument('--bias', default=False, type=bool)
    parser.add_argument('--no_compile', action='store_true') # if true then model is not compiled 
    
    parser.add_argument('--init_value', default=1.0, type=float)
    parser.add_argument('--memory', default=-1, type=int) # if negative, standard causal attention is applied


    # opt params
    # parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--num_iters', default=3000, type=int)

    parser.add_argument('--batch_size', default=64, type=int) #50
    parser.add_argument('--lr', default=1e-4, type=float) #2e-3
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)
    parser.add_argument('--scheduler', default='step', choices=['linear', 'cos', 'none', 'step'])
    parser.add_argument('--opt', default='adamw', choices=['adamw', 'sgd'])

    parser.add_argument('--warmup_percent', default=0.02, type=float)
    parser.add_argument('--div_factor', default=1e0, type=float)
    parser.add_argument('--final_div_factor', default=1e0, type=float)
    parser.add_argument('--grad_clip', default=0.0, type=float) # default value is 1.0 in NanoGPT

    parser.add_argument('--num_ckpt', default=150, type=int) # in iterations
    
    # parser.add_argument('--init', default='zero', choices=['zero', 'random'])
    
    # log params
    parser.add_argument('--results_base_folder', default="./exps", type=str) 
    
    
    args = parser.parse_args()

    
    return args

    


# ------------------------- logger ------------------------------------------------------

def raise_and_log_error(err_str, err_type, logger=None):
    if logger is not None:
        logger.error(err_str)

    if err_type == 'value':
        raise ValueError(err_str)
    if err_type == 'implement':
        raise NotImplementedError(err_str)
    else:
        raise ValueError('error type not implemented :))')
    
class Logger:
    def __init__(self, log_file=None, level=logging.INFO, log_to_console=True):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        # order of levels: debug -> info -> warning -> error -> critical

        self.log_path = log_file 

        # Formatter for log messages
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler (if a file is specified)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='a') # if file exists, append, don't overwrite
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Console handler (if enabled)
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def get_log_directory(self):
        """Returns the directory where the log file is stored."""
        if self.log_path:
            return os.path.dirname(os.path.abspath(self.log_file))
        return None  # No log file specified

    def get_log_path(self):
        """Returns the path where the log file is stored."""
        return self.log_path
    
# ------------ exp_name formatter -------------------------------------------
def format_hyperparams(precision=2, skip_keys=None, **kwargs):

    def format_value(value):
        if isinstance(value, float):
            return f"{value:.{precision}f}".replace(".", "_")  # Convert to sci notation & replace dot
        return str(value)  # Keep non-floats as is
    
    formatted_parts = []
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:  # Only add the key if True
                formatted_parts.append(f"{key}")
            continue  # Skip adding the key if False
        
        # Skip the key if it's in the skip_keys list
        if key in skip_keys:
            formatted_parts.append(f"{format_value(value)}")  # Only add the value, no key
        else:
            formatted_parts.append(f"{key}{format_value(value)}")  # Add both key and value
    
    return "_".join(formatted_parts)

def make_json_compatible(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, list):
        return [make_json_compatible(item) for item in obj]
    # elif isinstance(obj, (float, int, str, bool)) or obj is None:
        # return obj
    print(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

