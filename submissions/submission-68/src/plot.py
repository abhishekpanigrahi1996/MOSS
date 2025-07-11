import torch
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt

import gc
import os

import argparse
import json

import sys

from .utils import Logger, format_hyperparams

from argparse import Namespace

gc.collect()


def plot_(save_dict, job_path=None, save_path=None):
    args = Namespace(**save_dict['args'])
    if job_path is not None:
        ckpt_path = os.path.join(job_path, args.ckpt_path)
    else:
        ckpt_path = args.ckpt_path

    nrows, ncols = (2,3)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4* ncols, 3 * nrows), dpi=100)
    fig.subplots_adjust(wspace=0.3, hspace=0.4) 
    font_size = {'y':30, 'title':20, 'suptitle':50, 'tick':25, 'legend':8}
    linewidth=1.5

    metrics = save_dict['metrics']
    tmpl_type = save_dict['tmpl_type']

    mode_list = ['in_dist', 'out_dist'] if args.ood_frac > -1 else ['in_dist']
    style_dict = {'in_dist': '-', 'out_dist': '--'}
    color_list = ['tab:blue', 'tab:red', 'tab:green', 'black']

    # fig.suptitle(f'V{args.V}-M{args.M}-order{args.order}-L{args.n_layer}H{args.n_head}D{args.n_embd}-seqlen{args.seq_len}-mc{args.num_chain_tmpl}-pos{args.num_pos_tmpl}-bi{args.bi_pos}-chainperpos{args.chain_per_pos}-oodfrac{args.ood_frac}')
    
    for mode in mode_list:
        axs[1,0].plot(metrics['iter'], metrics[mode]['loss'], 
                    linewidth=linewidth, marker='', label=f'{mode}', linestyle=style_dict[mode], color=color_list[0])
        axs[1,0].set_yscale('log')
        axs[1,0].set_title(f'validation NTP loss')
        axs[1,0].legend(fontsize=font_size['legend'])

        axs[0,0].plot([args.batch_size * it / 1000 for it in metrics['iter']], metrics[mode]['kl_masked_completion_GT'], 
                    linewidth=linewidth, marker='', label=f'masked', linestyle=style_dict[mode], color=color_list[0])
        axs[0,0].set_title(r'$\mathtt{Loss}_{\mathrm{stat}}$')
        
        
        
    # for pos templates --------------------------------------------------------
    if (args.num_pos_tmpl > 0) or (args.bi_pos=="fixed"):
        pos_acc = "combined_non_kb_and_kb_at_pos"
        pos_acc = {mode: (np.array(metrics[mode]["non_kb_token_count_outside_reserved"]) /24 +
                            np.array(metrics[mode]["at_pos_is_kb_rate"])) / 2
                    for mode in mode_list}
        
        for mode in mode_list:

            axs[0,1].plot([args.batch_size * it / 1000 for it in metrics['iter']], metrics[mode]['at_pos_is_bi_rate'], 
                            linewidth=linewidth, linestyle=style_dict[mode], marker='', color=color_list[0],)# label=r'$\mathtt{Acc}_{\mathrm{fact}}$')
            axs[0,1].set_title(r'$\mathtt{Acc}_{\mathrm{fact}}$')
            axs[0,1].set_ylim([-0.05,1.05])
            
            
            axs[0,2].plot([args.batch_size * it / 1000 for it in metrics['iter']], pos_acc[mode], 
                        linewidth=linewidth, linestyle=style_dict[mode], marker='', color=color_list[0],)# label=r'$\mathtt{Acc}_{\mathrm{pos}}$')
            axs[0,2].set_title(r'$\mathtt{Acc}_{\mathrm{pos}}$')
            axs[0,2].set_ylim([-0.05,1.05])


                

        # HEATMAPS
        if tmpl_type == 'mc-pos':
            if save_dict['num_chain_tmpl'] == 1:
                # print(np.array(metrics['kp_tmpl_count']['train']).shape)
                g = sns.heatmap(np.array(metrics['kp_tmpl_count']['train'])[0,:,:],
                                ax=axs[1,2], annot=True, cmap="viridis", annot_kws={"size": 3})
                lbl = 'position index'
                g.set_facecolor('grey')
                axs[1,2].set_ylabel(lbl)


            elif save_dict['num_pos_tmpl'] == 1:
                g = sns.heatmap(np.array(metrics['kp_tmpl_count']['train'])[:,0,:],
                                ax=axs[1,2], annot=True, cmap="viridis", annot_kws={"size": 3})
                lbl = 'mc index'
                g.set_facecolor('grey')
                axs[1,2].set_ylabel(lbl)



            if save_dict['num_chain_tmpl'] == 1:
                g = sns.heatmap(np.array(metrics['kp_tmpl_count']['out_dist'])[0,:,:],
                                ax=axs[1,1], annot=True, cmap="viridis", annot_kws={"size": 3})
                lbl = 'position index'
                g.set_facecolor('grey')

                axs[1,1].set_ylabel(lbl)

            elif save_dict['num_pos_tmpl'] == 1:
                g = sns.heatmap(np.array(metrics['kp_tmpl_count']['out_dist'])[:,0,:],
                                ax=axs[1,1], annot=True, cmap="viridis", annot_kws={"size": 3})
                lbl = 'mc index'
                g.set_facecolor('grey')

                axs[1,1].set_ylabel(lbl)




        elif tmpl_type in ['mc', 'pos']:

            g = sns.heatmap(np.array(metrics['kp_tmpl_count']['train']),
                            ax=axs[1,2], annot=True, cmap="viridis", annot_kws={"size": 3})
            g.set_facecolor('grey')

            lbl = 'mc index' if tmpl_type=='mc' else 'position index'
            axs[1,2].set_ylabel(lbl)


            g = sns.heatmap(np.array(metrics['kp_tmpl_count']['out_dist']),
                            ax=axs[1,1], annot=True, cmap="viridis", annot_kws={"size": 3})
            g.set_facecolor('grey')
            
            lbl = 'mc index' if tmpl_type=='mc' else 'position index'
            axs[1,1].set_ylabel(lbl)

        
        axs[1,2].set_title('in-dist exposure mask')
        axs[1,2].set_xlabel('knowledge pair')
        axs[1,2].invert_yaxis()
        
        axs[1,1].set_title('out-dist exposure mask')
        axs[1,1].set_ylabel('template index')
        axs[1,1].set_xlabel('knowledge pair')
        axs[1,1].invert_yaxis()






    name = f'MC{args.num_chain_tmpl}Pos{args.num_pos_tmpl}-V{args.V}M{args.M}-L{args.n_layer}H{args.n_head}D{args.n_embd}-seqlen{args.seq_len}-oodfrac{int(args.ood_frac*10)}.pdf'
    if save_path is not None:
        file_name=os.path.join(save_path,name)
        print(file_name)
    else:
        file_name = os.path.join(ckpt_path, name)

    fig.savefig(file_name, dpi=1200, transparent=False)

