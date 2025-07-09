'''
This file was used to run the experiments provided in Appendix B of the transience paper:
Namely, those on fixed omniglot embeddings, extracted with omni_features_extract.py
'''
import submitit
from copy import deepcopy

from main_utils import create_parser
from main import run_with_opts
import opto_ as opto

SAVE_FOLDER = "TODO"
INIT_SEED=5
MAIN_RUN_ITERS = 40960000 # 2048 * 20000
LONG_RUN_ITERS = 409600000

parser = create_parser()
opto.add_args_to_parser(parser)

run_name = 'icl'
# Note default data file is for 1600 classes
lin_opts = parser.parse_args(
    [
        '--use_wandb',
        '--no_embed',
        '--no_unembed',
        '--not_causal',
        '--exclude_query_token',
        '--no_softmax',
        '--balanced_classes_queries',
        '--init_rescale', '0.002',
        '--classification',
        '--pos_embedding_type', 'none',
        '--n_labels', '5',
        '--train_context_len', '100',
        '--X_dim', '13',
        '--eval_context_len', '100',
        '--eval_n_samples', '512',
        '--no_norm',
        '--optimizer', 'adam',
        '--grad_clip_val', '0.001',
        '--depth', '1',
        '--train_iters', str(MAIN_RUN_ITERS),
        '--train_bs', '2048',
        '--eval_every', '204800',
        '--lr', '0.0005',
        '--d_model', '18',
        '--num_heads', '1',
        '--run', 'lin_c100_lr0.0005',
        '--ckpt_every', '204800',
        '--init_seed', str(INIT_SEED),
        '--base_folder', SAVE_FOLDER,
        '--raw_name',
        '--no_proj_bias'
    ]
)

sfmx_opts = deepcopy(lin_opts)
sfmx_opts.lr = 0.002
sfmx_opts.init_rescale = 0.002
sfmx_opts.no_softmax = False
sfmx_opts.grad_clip_val = 1.0

executor = submitit.AutoExecutor(folder=f"{SAVE_FOLDER}/log_submitit")
executor.update_parameters(
  name=run_name,
  gpus_per_node=1,
  tasks_per_node=1,
  cpus_per_task=10,
  timeout_min=4320,
  slurm_partition="TODO",
  slurm_array_parallelism=8,
)

all_opts = []

X_dim = 2
cont_len = 100
n_labels = 5

all_opts.append(deepcopy(sfmx_opts))
all_opts[-1].run = f'sfmx_q_xdim_{X_dim}_nlab_{n_labels}_clen_{cont_len}_trained_lr_0006_niter_20000'
all_opts[-1].train_seed = 11
all_opts[-1].init_seed = 8
all_opts[-1].n_labels = n_labels
all_opts[-1].X_dim = X_dim
all_opts[-1].train_context_len = cont_len
all_opts[-1].eval_context_len = cont_len
all_opts[-1].d_model = all_opts[-1].X_dim + all_opts[-1].n_labels
all_opts[-1].lr = 0.0006
all_opts[-1].train_iters = LONG_RUN_ITERS*10

all_opts.append(deepcopy(lin_opts))
all_opts[-1].run = f'lin_q_xdim_{X_dim}_nlab_{n_labels}_clen_{cont_len}_trained'
all_opts[-1].train_seed = 11
all_opts[-1].init_seed = 8
all_opts[-1].X_dim = X_dim
all_opts[-1].n_labels = n_labels
all_opts[-1].eval_context_len = cont_len
all_opts[-1].train_context_len = cont_len
all_opts[-1].d_model = all_opts[-1].X_dim + all_opts[-1].n_labels
all_opts[-1].train_iters = LONG_RUN_ITERS
all_opts[-1].lr = 0.00005

X_dim = 3
cont_len = 100
n_labels = 5

all_opts.append(deepcopy(sfmx_opts))
all_opts[-1].run = f'sfmx_q_xdim_{X_dim}_nlab_{n_labels}_clen_{cont_len}_trained_lr_00003_niters_20000'
all_opts[-1].train_seed = 11
all_opts[-1].init_seed = 8
all_opts[-1].n_labels = n_labels
all_opts[-1].X_dim = X_dim
all_opts[-1].train_context_len = cont_len
all_opts[-1].eval_context_len = cont_len
all_opts[-1].d_model = all_opts[-1].X_dim + all_opts[-1].n_labels
all_opts[-1].lr = 0.00003
all_opts[-1].train_iters = LONG_RUN_ITERS*10

all_opts.append(deepcopy(lin_opts))
all_opts[-1].run = f'lin_q_xdim_{X_dim}_nlab_{n_labels}_clen_{cont_len}_trained'
all_opts[-1].train_seed = 11
all_opts[-1].init_seed = 8
all_opts[-1].X_dim = X_dim
all_opts[-1].n_labels = n_labels
all_opts[-1].eval_context_len = cont_len
all_opts[-1].train_context_len = cont_len
all_opts[-1].d_model = all_opts[-1].X_dim + all_opts[-1].n_labels
all_opts[-1].train_iters = LONG_RUN_ITERS
all_opts[-1].lr = 0.00005

X_dim = 5
cont_len = 100
n_labels = 5

all_opts.append(deepcopy(sfmx_opts))
all_opts[-1].run = f'sfmx_q_xdim_{X_dim}_nlab_{n_labels}_clen_{cont_len}_trained_lr_0001_niter_20000'
all_opts[-1].train_seed = 11
all_opts[-1].init_seed = 8
all_opts[-1].n_labels = n_labels
all_opts[-1].X_dim = X_dim
all_opts[-1].train_context_len = cont_len
all_opts[-1].eval_context_len = cont_len
all_opts[-1].d_model = all_opts[-1].X_dim + all_opts[-1].n_labels
all_opts[-1].lr = 0.0001
all_opts[-1].train_iters = LONG_RUN_ITERS*10

all_opts.append(deepcopy(lin_opts))
all_opts[-1].run = f'lin_q_xdim_{X_dim}_nlab_{n_labels}_clen_{cont_len}_trained'
all_opts[-1].train_seed = 11
all_opts[-1].init_seed = 8
all_opts[-1].X_dim = X_dim
all_opts[-1].n_labels = n_labels
all_opts[-1].eval_context_len = cont_len
all_opts[-1].train_context_len = cont_len
all_opts[-1].d_model = all_opts[-1].X_dim + all_opts[-1].n_labels
all_opts[-1].train_iters = LONG_RUN_ITERS
all_opts[-1].lr = 0.00005

X_dim = 10
cont_len = 100
n_labels = 5

all_opts.append(deepcopy(sfmx_opts))
all_opts[-1].run = f'sfmx_q_xdim_{X_dim}_nlab_{n_labels}_clen_{cont_len}_trained_lr_0005_niter_20000'
all_opts[-1].train_seed = 11
all_opts[-1].init_seed = 8
all_opts[-1].n_labels = n_labels
all_opts[-1].X_dim = X_dim
all_opts[-1].train_context_len = cont_len
all_opts[-1].eval_context_len = cont_len
all_opts[-1].d_model = all_opts[-1].X_dim + all_opts[-1].n_labels
all_opts[-1].lr = 0.0005
all_opts[-1].train_iters = LONG_RUN_ITERS*10

all_opts.append(deepcopy(lin_opts))
all_opts[-1].run = f'lin_q_xdim_{X_dim}_nlab_{n_labels}_clen_{cont_len}_trained'
all_opts[-1].train_seed = 11
all_opts[-1].init_seed = 8
all_opts[-1].X_dim = X_dim
all_opts[-1].n_labels = n_labels
all_opts[-1].eval_context_len = cont_len
all_opts[-1].train_context_len = cont_len
all_opts[-1].d_model = all_opts[-1].X_dim + all_opts[-1].n_labels
all_opts[-1].train_iters = LONG_RUN_ITERS
all_opts[-1].lr = 0.0000

  
# run all the jobs
jobs = executor.map_array(run_with_opts, all_opts)