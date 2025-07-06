import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import math
import sys
import time

from collections import Counter
from contextlib import nullcontext
from collections import defaultdict


from .utils import Logger
from .metrics import *


def count_kps_tmpls(kps, tmpls, kp_base, tmpl_type, num_chain_tmpl, num_pos_tmpl):
    """
    Count how often each knowledge pair in kp_base appears overall (count_array)
    and how often it appears per template (count_matrix), where:
      - tmpl_type == 'mc': template is (mc_id, -1)
      - tmpl_type == 'pos': template is (-1, pos_id)
      - tmpl_type == 'mc-pos': template is (mc_id, pos_id)
    """
    if not kps:
        return 0, 0

    # Count overall appearances of kps
    pair_counts = Counter(kps)
    count_array = np.array([pair_counts.get(pair, 0) for pair in kp_base])
    pair_idx = {pair: i for i, pair in enumerate(kp_base)}

    # Count (template, kp) pairs
    pair_template_counts = Counter(zip(kps, tmpls))

    if tmpl_type == 'mc':
        count_matrix = np.zeros((num_chain_tmpl, len(kp_base)), dtype=int)
        for (kp, (mc, _)), count in pair_template_counts.items():
            count_matrix[mc, pair_idx[kp]] = count

    elif tmpl_type == 'pos':
        count_matrix = np.zeros((num_pos_tmpl, len(kp_base)), dtype=int)
        for (kp, (_, pos)), count in pair_template_counts.items():
            count_matrix[pos, pair_idx[kp]] = count

    elif tmpl_type == 'mc-pos':
        count_matrix = np.zeros((num_chain_tmpl, num_pos_tmpl, len(kp_base)), dtype=int)
        for (kp, (mc, pos)), count in pair_template_counts.items():
            count_matrix[mc, pos, pair_idx[kp]] = count
    else: 
        count_matrix = None

    return count_array, count_matrix

def train(model, opt, generator, scheduler, extra_args, logger):
    num_ckpt = extra_args.num_ckpt

    num_iters = extra_args.num_iters
    batch_size = extra_args.batch_size
    context_length = extra_args.context_length

    device = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    
    type_ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(
        device_type=device, dtype=torch.float16)  # extra_args.dtype) #changed!
    
    num_pos_tmpl = generator.num_pos_tmpl
    use_pos_tmpl = (num_pos_tmpl != -1)
    num_chain_tmpl = generator.num_chain_tmpl
    tmpl_type = generator.tmpl_type

    
    ood_frac = extra_args.ood_frac
    use_ood = ood_frac != -1
    chain_per_pos = extra_args.chain_per_pos
    ood_struct_eval = extra_args.ood_struct_eval


    # logger.info('-'*200)
    # logger.info(f'template type {tmpl_type}')
    # logger.info(f'using {num_pos_tmpl} position templates, {num_chain_tmpl} mc templates.')
    # logger.info(f'ood_frac = {ood_frac}, number of chains per positions for in-dist mask {chain_per_pos}')
    
    
    # -------------------------------------------------------------------------------------------
    itr = 0

    V = generator.V         # vocab_size
    M = generator.M          # number of knowledge pairs
    knowledge_base, _ = generator.get_knowledge_base()
    

    ckpts = {
            'in_dist': defaultdict(list), # results 
            'iter': [],
            'lr': [],
            'kp_count': {'train': None, 'in_dist': None, 'out_dist':None, "out_struct":None},
            'kp_tmpl_count': {'train': None, 'in_dist': None, 'out_dist':None, "out_struct":None },
            'pos_chain_mask':  generator.pos_chain_mask,
        }
    
    if use_ood:
        ckpts['out_dist'] = defaultdict(list)
    if ood_struct_eval:
        ckpts['out_struct'] = defaultdict(list)
      
    for mode in ckpts['kp_count'].keys():
        ckpts['kp_count'][mode] = np.zeros((M,))
        if tmpl_type in ['mc', 'pos']:
            num_tmpl = num_chain_tmpl if tmpl_type == 'mc' else num_pos_tmpl
            ckpts['kp_tmpl_count'][mode] = np.zeros((num_tmpl, M))
        if tmpl_type == 'mc-pos':
            ckpts['kp_tmpl_count'][mode] = np.zeros((num_chain_tmpl, num_pos_tmpl, M))

    # ------------------------------------------------------------------------------------------------------------
    iter_list = np.logspace(0, np.log10(num_iters), num=num_ckpt, dtype=int)
    iter_list = np.concatenate(([0], iter_list, [num_iters-1, num_iters]))
    iter_list = np.unique(iter_list).tolist()
    # logger.info(f'list of evalutation iters: {iter_list}')

    
    # --------------------------- data for validation -----------------------------------------------------------
    # for val loss computation
    batch_val = {}
    batch_val['in_dist'] = generator.generate_batch(num_sequences=extra_args.test_size, max_length=context_length, ood=False, ood_struct=False)
    if use_ood:
        batch_val['out_dist'] = generator.generate_batch(num_sequences=extra_args.test_size, max_length=context_length, ood=True, ood_struct=False)
    if ood_struct_eval:
        batch_val['out_struct'] = generator.generate_batch(num_sequences=extra_args.test_size, max_length=context_length, ood=False, ood_struct=True)
    
    
    # for completion and knowledge eval
    batch_gn = {}
    
    batch_gn['in_dist'] = generator.generate_batch(num_sequences=extra_args.test_size, max_length=context_length, only_ai=True, ood=False, ood_struct=False)
    
    batch_gn['out_dist'] = generator.generate_batch(num_sequences=extra_args.test_size, max_length=context_length, only_ai=True, ood=True, ood_struct=False)
    if ood_struct_eval:
        batch_gn['out_struct'] = generator.generate_batch(num_sequences=extra_args.test_size, max_length=context_length, only_ai=True, ood=False, ood_struct=True)
    
    # save the number of times each template/knowledge piece appears
    for mode in ['in_dist', 'out_dist', 'out_struct']:
        if mode not in batch_gn.keys():
            continue
        
        kp_count, kp_tmpl_count = count_kps_tmpls(batch_gn[mode]['kps'], batch_gn[mode]['tmpls'], knowledge_base, tmpl_type, num_chain_tmpl, num_pos_tmpl)
        ckpts['kp_count'][mode] = kp_count
        ckpts['kp_tmpl_count'][mode] = kp_tmpl_count

    dt = 0
    ckpt_count = 0 

    for itr in range(num_iters):

        # generate a batch of data for training update
        batch = generator.generate_batch(num_sequences=batch_size, max_length=context_length, ood=False, ood_struct=False)

        # evaluate
        if itr in iter_list:
            model.eval()

            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
            ckpts['iter'].append(itr)
            ckpts['lr'].append(current_lr)
            
    
            t2 = time.time()

            pos_tmpls = generator.pos_tmpls if use_pos_tmpl else None
            transition_dict = generator.transition_mat_dict
            stats = eval(model, batch_val, batch_gn, extra_args, logger, type_ctx, device, pos_tmpls=pos_tmpls,
                        transition_dict=transition_dict)
            t3 = time.time()
            dt_eval = t3 - t2

            for mode in stats.keys():
                for key, value in stats[mode].items():
                    ckpts[mode][key].append(value)

            print_string = f"{itr} [in-dist] loss={stats['in_dist']['loss']:.3f}"
            print_string += f" [time per itr] {dt*1000:.2f}ms"
            print_string += f" [time per eval itr] {dt_eval*1000:.2f}ms"
            
            if scheduler is not None:
                print_string += f" [lr] {current_lr:.5f}"

            if ckpt_count % 5 == 0:
                logger.info(print_string)

            ckpt_count += 1


        # train
        model.train()
        
        # save the number of times each template/knowledge piece appears

        kp_count, kp_tmpl_count = count_kps_tmpls(batch['kps'], batch['tmpls'], knowledge_base, tmpl_type, num_chain_tmpl, num_pos_tmpl)
        ckpts['kp_count']['train'] += kp_count
        if kp_tmpl_count is not None:
            ckpts['kp_tmpl_count']['train'] += kp_tmpl_count

        # train
        t0 = time.time()
        x, y, attn_mask = batch['x'].to(device), batch['y'].to(device), batch['attn_mask'].to(device)
        with type_ctx:
            outputs = model(x, targets=y, attn_mask=attn_mask, get_logits=True, 
                            pad_idx = extra_args.special_toks['PAD'], loss_reduction='mean')
        loss = outputs['loss']
        loss.backward()

        if extra_args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)
        
        opt.step()
        if scheduler is not None:
            scheduler.step()
        opt.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1-t0

        itr += 1

    return ckpts



def eval(model, batch_val, batch_gn, extra_args, logger, type_ctx = nullcontext(), device='cpu', 
         pos_tmpls=None, transition_dict=None):

    stats = {}

    for mode in batch_gn.keys():

        batch = batch_val[mode]
    
        x, y, attn_mask = batch['x'].to(device), batch['y'].to(device), batch['attn_mask'].to(device)
        
        with type_ctx:
            outputs = model(x, targets=y, attn_mask=attn_mask, get_logits=True, 
                            pad_idx = extra_args.special_toks['PAD'], loss_reduction='mean')

        kp_list = batch['kps']
        tmpl_list = batch['tmpls']

        if tmpl_list[0] == (-1,-1):
            tmpl_type = None
        elif tmpl_list[0][0] == -1:
            tmpl_type = 'pos'
        elif tmpl_list[0][1] == -1:
            tmpl_type = 'mc'
        else:
            tmpl_type = 'mc-pos'

        _, L, V = outputs['logits'].shape
        
        # loss
        loss = outputs['loss']
        stats_mode = {'loss': loss.item()}


        
        x_gn = batch_gn[mode]['x'].to(device)
        
        if tmpl_type in ['pos', 'mc-pos']:
            tmpl_gen = batch_gn[mode]['tmpls']
            tmpl_pos_list = [tmpl[1] for tmpl in tmpl_gen]
            pos_list = [pos_tmpls[idx][1] for idx in tmpl_pos_list]
        else:
            pos_list = None
        
        gen_metrics =  generate_and_evaluate(model, x_gn, 
                                            batch_gn[mode]['kps'], 
                                            knowledge_vocab_list=extra_args.knowledge_tokens, 
                                            V=V, pos_tmpls=pos_list,
                                            max_gen_len=(L//2),
                                            transition_dict=transition_dict, tmpl_list=tmpl_gen)
        stats_mode.update(gen_metrics)
   
        stats[mode] = stats_mode

  
    return stats


