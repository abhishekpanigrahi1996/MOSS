import os
import json
import numpy as np
import sys
import math
import time
import inspect
import logging

from itertools import product
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import StepLR

from utils import get_args, format_hyperparams, make_json_compatible, Logger, raise_and_log_error
from data import MarkovKnowledgeGenerator
from model_base import GPTBase
from train_utils import train
from plot import plot_



def dict_format():
    return {
        'knowledge_pairs': None,
        'knowledge_tokens': None,
        'chain_tokens': None,
        'num_chain_tmpl': None,
        # 
        'kp_tmpl_count': None,
        'kp_count': None,
        # 
        'loss_list': {'in_dist': [], 'out_dist': []},
        'kp_acc_list': {'in_dist': [], 'out_dist': []},
        'KL_list': {'in_dist': [], 'out_dist': []},
        'iter_list': [],
        'lr_list': [],
    }


def main(args):
    
    # ------------------------------------- logging setup --------------------------------------------------------------------
    # TODO: fix the exp names......... reflect it in the save files too
    exp_name_dict ={
                    'V': args.V,
                    'M': args.M,
                    'seq': args.seq_len,
                    'mc': args.num_chain_tmpl,
                    'pos': args.num_pos_tmpl,
                    'L': args.n_layer,
                    'H': args.n_head,
                    'd': args.n_embd,
                    }
    skip_keys = ['data',]
    exp_name = format_hyperparams(precision=1, skip_keys=skip_keys, **exp_name_dict)
    print(exp_name)

    ckpt_path = os.path.join(args.results_base_folder, exp_name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    log_path = os.path.join(ckpt_path, 'log.txt')
    logger = Logger(log_file=log_path, level=logging.DEBUG, log_to_console=False)
    args.log_path = log_path
    args.ckpt_path = ckpt_path

    # saving hyper-params:
    filtered_args = {k: v for k, v in vars(args).items() if k not in ['device', 'dtype']}
    hp_path = os.path.join(args.results_base_folder, 'hyperparams.json')
    with open(hp_path, 'w') as json_file:
        json.dump(filtered_args, json_file, indent=4, default=make_json_compatible)  
    logger.info(f'logging setup done: saving in f{ckpt_path}')


    # ------------------------------------- exp setup --------------------------------------------------------------------
    # device
    args.device = torch.device(args.device)
    device = "cuda" if "cuda" in str(args.device) else "cpu"
    if device == "cuda":
        torch.cuda.set_device(args.device)

    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True
    
    save_dict = {}

    # ------------------------ data setup ---------------------------------------------------------------------------------------
    order = args.order
    V = args.V          # vocab_size
    M = args.M          # number of knowledge pairs
    seq_len = args.seq_len
    initial = args.initial

    # special tokens
    if args.skip_spectok:
        PAD = None
        args.vocab_size = V 
        args.special_toks = {'PAD':-1}
    else:
        PAD = V
        BOS = V + 1
        EOS = V + 2
        args.vocab_size = V + 3
        args.special_toks = {'PAD': PAD, 'BOS':BOS, 'EOS':EOS}


    MCGenerator = MarkovKnowledgeGenerator(V=V, M=M, order=order, seq_length=seq_len, special_toks=args.special_toks, logger=logger,
                                        insertion_mode=args.insertion_mode, device=device, skip_sepctok=args.skip_spectok,
                                        num_chain_tmpl=args.num_chain_tmpl, initial=initial, 
                                        num_pos_tmpl=args.num_pos_tmpl, bi_pos = args.bi_pos,
                                        ood_frac=args.ood_frac, chain_per_pos=args.chain_per_pos,
                                        )
    
    save_dict['knowledge_pairs'], save_dict['knowledge_tokens'] = MCGenerator.get_knowledge_base()
    save_dict['chain_tokens'], save_dict['num_chain_tmpl'] = MCGenerator.get_MC_props()
    args.chain_tokens = save_dict['chain_tokens']
    args.knowledge_tokens = save_dict['knowledge_tokens']
    save_dict['mem_dict'] = MCGenerator.mem_dict
    save_dict['tmpl_type'] = MCGenerator.tmpl_type
    save_dict['num_chain_tmpl'] = MCGenerator.num_chain_tmpl
    save_dict['num_pos_tmpl'] = MCGenerator.num_pos_tmpl

    # print('CHAIN:', MCGenerator.transition_mat_dict)

    logger.info(f'data generator loaded\n' + "\n".join([
                                            f"knowledge base: {save_dict['knowledge_pairs']}",
                                            f"knowledge tokens: {save_dict['knowledge_tokens']}",
                                            f"chain tokens: {save_dict['chain_tokens']}",
                                            f"number of chains: {save_dict['num_chain_tmpl']}",
                                            '-'*100])
                )
    if args.num_pos_tmpl > 0:
        save_dict['pos_tmpl'] = MCGenerator.pos_tmpls
        save_dict['indist_mask'] = MCGenerator.in_dist_mask
        pos_dict = {i: pair for i,pair in enumerate(save_dict['pos_tmpl'])}
        logger.info(f'position templates: \n{pos_dict}')

    # ----------------------- model setup ------------------------------------------------------------------------------------
    args.no_tying = True
    model = GPTBase(args, pad_idx=PAD).to(args.device) 
    group_specs = model.get_parameter_group_specs() 
    
    # changed this part: count the params directly, not from distributed back-end
    logger.info('model loaded')
    num_params = model.get_num_params()
    logger.info(f'number of optimized params: {num_params/1e3:.2f}K')
    logger.debug(model)
    logger.info('-'*100)

    # ----------------------- opt setup -------------------------------------------------------------------------------------------
    if args.opt == 'adamw':
        use_fused = (device == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        logger.info('using fused AdamW: {}'.format(use_fused))

        extra_args = dict(fused=True) if use_fused else dict()
        opt = torch.optim.AdamW(group_specs, lr=args.lr, betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay, **extra_args)
    else:
        opt = torch.optim.SGD(group_specs, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    logger.info('Loading scheduler {}'.format(args.scheduler))
    if args.scheduler != 'none':
        if args.scheduler in ['cos', 'linear']:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt, max_lr=args.lr, total_steps=args.num_iters, 
                                                            pct_start=args.warmup_percent, anneal_strategy=args.scheduler, 
                                                            cycle_momentum=False, div_factor=args.div_factor, final_div_factor=args.final_div_factor)
        elif args.scheduler == 'step':
            scheduler = StepLR(opt, step_size=3000, gamma=1.0) # gamma=1.0 -> constant learning rate
        else:
            err_str = f"Unknown scheduler type: {args.scheduler}."
            raise_and_log_error(err_str=err_str, err_type='implement', logger=logger)
    else:
        scheduler = None


    # ----------------------- training setup -----------------------------------------------------------------------------------------
    logger.info(f'\nTraining model with config: \n{vars(args)}\n')
    logger.info('-'*100)
    t_start = time.time()
    ckpts = train(model, opt, MCGenerator, scheduler, args, logger)

    # saving and final logs:
    logger.info('Training done')

    # ----------------------- save --------------------------------------------------------------------------------------------------
    save_dict['metrics'] = ckpts
    
    # mode_list = ckpts['kp_count'].keys()
    # for key, mode in product(['kp_count', 'kp_tmpl_count'], mode_list):
    #     # save_dict[key][mode] = ckpts[key][mode]
    #     logger.info(f'{key} - {mode}: {save_dict['metrics'][key][mode]}')

    # if args.num_pos_tmpl > 0:
    #     logger.info(f'in-dist mask: {save_dict['indist_mask']}')

    # saving and final logs:                
    args.device = None
    args.dtype = None
    save_dict['args'] = vars(args)

    # logger.debug(save_dict)
    
    with open(f"{ckpt_path}/train_results.json", "w") as fs:
        json.dump(save_dict, fs, indent=4, default=make_json_compatible)

    logger.info('saving variables done!')

    # ----------------------- plot --------------------------------------------------------------------------------------------------
    plot_(save_dict)
    logger.info('plotting...\n')

    logger.info(f'Total run time: {(time.time()-t_start)/60}min')
    logger.info('END'+'_'*200)




if __name__ == "__main__":
    args = get_args()
    main(args)