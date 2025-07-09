'''
This file all the argparse arguments for standard training, as well as
some util methods on top of them (to set defaults etc.)
'''

import argparse
import json

import jax
import equinox as eqx
import optax

import models


def create_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config_from_file', default=None, type=str, help="Config filename. Note that EVERYTHING will be overridden/any other args you pass in will be ignored. Use this option only when you essentially want to duplicate an experiment.")

  parser.add_argument('--base_folder', default='./runs', type=str, help='Path to base folder to store runs')
  parser.add_argument('--use_wandb', action='store_true', help="Defaults to false. If specified, log to wandb.")
  parser.add_argument('--raw_name', action='store_true', help='Defaults to false. When false, the datetime is prepended to the runname for ease of distinguishing. If specified, we will not prepend the datetime.')
  parser.add_argument('--run', default='run', type=str, help="Run name")
  parser.add_argument('--disable_jit', action='store_true', help="If specified, disable jitting. This is redundant to setting JAX_DISABLE_JIT=1 as an env variable.")


  # Model params
  parser.add_argument('--depth', default=2, type=int, help="Transformer depth.")
  parser.add_argument('--not_causal', action='store_true', help="Whether to make transformer causal. Defaults to causal.")
  parser.add_argument('--exclude_query_token', action='store_true', help="Whether to use attention with masked query token. Defaults to no.")
  parser.add_argument('--init_rescale', default=1.0, type=float, help="How much to scale all weight parameter initializations by before training.")
  parser.add_argument('--pos_embedding_type', default='rope', type=str, choices=['none', 'ape', 'rope', 'sinusoidal'], help="What type of positional embedding to use. Defaults to 'rope' (rotary positional embeddings).")
  parser.add_argument('--sin_time', type=float, default=10000.0, help="What the timescale used to calculate sinusoidal embeddings (for RoPE) should be.")
  parser.add_argument('--d_model', default=64, type=int, help="Transformer dimension.")
  parser.add_argument('--no_embed', action='store_true', help="Whether to do initial embedding.")
  parser.add_argument('--no_unembed', action='store_true', help="Whether to do unembedding in the end.")
  parser.add_argument('--num_heads', default=8, type=int, help="Number of heads per layer.")
  parser.add_argument('--mlp_ratio', default=None, type=float, help="Expansion factor to use in MLP layers. When set to None, the model becomes attention-only (no MLP layers are used).")
  parser.add_argument('--model_output_classes', default=None, type=int, help="How many output classes for the model. The default is a bit complicated but 'smart' hopefully. If fs_relabel is nonzero, the default is equal to the train fs_relabel. Otherwise, the default is the number of classes present in training (as having more classes would not get signal). There may still be a reason to make model_output_classes > # of train_classes (e.g. if you want to continue training the model on a new set of classes from a checkpoint/continual learning setup). Note if output classes is smaller than a class present in the data, that class will have loss 0 because of how jax.nn.one_hot behaves.")
  parser.add_argument('--no_norm', action='store_true', help="If specified, no norm layers.")
  parser.add_argument('--no_softmax', action='store_true', help="Whether to use softmax attention.")
  parser.add_argument('--normalize_output', action='store_true', help="Divide the output by context length.")
  parser.add_argument('--no_proj_bias', action='store_true', help="Whether to have the bias in the the projection of attention block.")
  
  # Training params
  parser.add_argument('--init_seed', default=5, type=int, help="Random seed for training")
  parser.add_argument('--train_seed', default=0, type=int, help="Random seed for training")
  parser.add_argument('--train_iters', default=int(1e5), type=int, help="# training iters")
  parser.add_argument('--train_bs', default=32, type=int, help="Train batch size. Note that train sequences will change if batch size changes.")
  parser.add_argument('--train_microbs', default=None, type=int, help="Train microbatch size. If specified, the train batch will be split into microbatches of the specified size, and gradients will be averaged before updating. NOTE: if microbatching is being used, the jitting needs to be modified (as we do not want to jit the full train step, as that would jit a big for loop).")
  parser.add_argument('--lr', default=1e-5, type=float, help="Learning rate")
  parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'amsgrad', 'sgd', 'warmup_decay', 'cosine'], help="Optimizer to use.")
  parser.add_argument('--weight_decay', default=0.0, type=float, help="Weight decay.")
  parser.add_argument('--grad_clip_val', default=10.0, type=float, help="Grad clipping value, gradients in between (-grad_clip_val, grad_clip_val)")
  parser.add_argument('--load_from_ckpt', default=None, type=str, help="Checkpoint file to load from")
  parser.add_argument('--load_from_ckpt_cfg', default=None, type=str, help="Config of checkpointed model to load. Overrides current model config params as otherwise equinox won't let us load in.")
  parser.add_argument('--ckpt_every', default=None, type=int, help="If specified, how often to checkpoint")
  parser.add_argument('--ckpt_sched', nargs='+', default=None, type=int, help="A list of iterations to checkpoint at. More easily specified with --ckpt_sched_file.")
  parser.add_argument('--ckpt_sched_file', default=None, type=str, help="If specified, a pickle file that when read in, provides a python array with checkpoint iterations. This array will be stored in ckpt_sched. Provides more flexibility for e.g. sampling more checkpoints around phase changes.")
  parser.add_argument('--warmup_steps', default=4_000, type=int, help="Number of warmup steps for learning rate schedule")
  parser.add_argument('--decay_steps', default=1_000_000, type=int, help="Number of decay steps for learning rate schedule")
  parser.add_argument('--mse_loss', action='store_true', help="Using mse loss.")
  

  # Data generation parameters
  parser.add_argument('--classification', action='store_true', help="If specified the data will be for classification task")
  parser.add_argument('--n_labels', default=2, type=int, help="Number of vectors / classes to use")
  parser.add_argument('--train_context_len', default=8, type=int, help="Train context length")
  parser.add_argument('--eval_context_len', default=6, type=int, help="Evaluation context length")
  parser.add_argument('--eval_n_samples', default=50, type=int, help="Number of evaluation samples (per batch)")
  parser.add_argument('--X_dim', default=10, type=int, help="Dimension of X vectors")
  parser.add_argument('--train_noise_scale', default=0.0, type=int, help="Noise to add on Ys during training")
  parser.add_argument('--eval_noise_scale', default=0.0, type=int, help="Noise to add on Ys during evaluation")
  parser.add_argument('--balanced_classes_queries', action='store_true', help="Use the same number of exemplars per class in training and sample queries from the same distribution")
  
  # eval params
  parser.add_argument('--eval_seed', default=1, type=int, help="Random seed for eval")
  parser.add_argument('--eval_iters', default=int(1e3), type=int, help="# eval sequences")
  parser.add_argument('--eval_every', default=None, type=int, help="How often to evaluate training")
  parser.add_argument('--eval_sched', nargs='+', default=None, type=int, help="A list of iterations to evaluate at. More easily specified with --eval_sched_file.")
  parser.add_argument('--eval_sched_file', default=None, type=str, help="If specified, a pickle file that when read in, provides a python array with eval iterations. This array will be stored in eval_sched.")
  parser.add_argument('--save_eval_data', default=None, type=str, help="If specified, file to store eval data to. Gets placed in runs/\{opts.run\}.")
  parser.add_argument('--load_eval_data', default=None, type=str, help="If specified, an h5 file that eval data can be loaded from. Should have a format compatible with save_eval_data")
  return parser


def check_opts(opts):
  assert opts.train_microbs <= opts.train_bs, 'Microbatch size must be smaller than batch size'
  assert opts.train_bs % opts.train_microbs == 0, 'Train batch size should be a multiple of microbatch size'
  assert opts.init_seed != opts.train_seed, 'Init and train seed must be diff'
  assert opts.train_seed != opts.eval_seed, 'Train and eval seed must be diff'
  assert opts.init_seed != opts.eval_seed, 'Init and eval seed must be diff'
  assert (opts.eval_every is not None) or (opts.eval_sched is not None) or (opts.eval_sched_file is not None), 'Some method of evaluating must be specified atm'
  assert (opts.eval_every is None) + (opts.eval_sched is None) + (opts.eval_sched_file is None) == 2, 'Exactly one way of eval iters should be specified'
  assert (opts.ckpt_every is None) + (opts.ckpt_sched is None) + (opts.ckpt_sched_file is None) >= 2, 'At most one way of ckpt iters should be specified'

def get_opts_from_json_file(fname):
  parser = create_parser()
  opts, unknown = parser.parse_known_args()
  with open(fname, 'r') as f:
    vars(opts).update(json.load(f))
  return opts


def get_opts_from_dict(opts_dict):
  return argparse.Namespace(**opts_dict)


def scale_model_init(model, scale=1.0):
  '''
  This method will scale all weight matrices down by a given factor
  '''
  is_linear = lambda x: isinstance(x, eqx.nn.Linear)
  get_weights = lambda m: [x.weight
                           for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                           if is_linear(x)]
  weights = get_weights(model)
  new_weights = [weight * scale for weight in weights]
  new_model = eqx.tree_at(get_weights, model, [weight * scale for weight in weights])
  return new_model


def get_model_from_opts(opts):
  opts.init_key = jax.random.PRNGKey(opts.init_seed)
  model = models.MergedTokensClassifier(
      example_shape=(opts.X_dim,),
      num_classes=opts.n_labels,
      do_embed=not opts.no_embed,
      do_unembed=not opts.no_unembed,
      embed_dim=opts.d_model,
      key=opts.init_key,
      depth=opts.depth,
      num_heads=opts.num_heads,
      mlp_ratio=opts.mlp_ratio,
      causal=(not opts.not_causal),
      exclude_query_token=opts.exclude_query_token,
      pos_embedding_type=opts.pos_embedding_type,
      sin_time=opts.sin_time,
      norm_layer=(eqx.nn.Identity if opts.no_norm else eqx.nn.LayerNorm),
      softmax=not opts.no_softmax,
      qk_scale=1.0 if opts.no_softmax else None,
      normalize_output=opts.normalize_output,
      proj_bias=not opts.no_proj_bias,
    )
  model = scale_model_init(model, opts.init_rescale)
  return model
  

def get_optimizer_from_opts(opts):
  if opts.optimizer == 'adam':
    """schedule = optax.warmup_cosine_decay_schedule(
      init_value=jnp.float32(0.0),
      peak_value=jnp.float32(0.001),
      warmup_steps=1000,
      decay_steps=opts.train_iters - 1000,
      end_value=jnp.float32(0.00005),
    )"""
    optimizer = optax.chain(
      optax.clip_by_global_norm(opts.grad_clip_val),
      optax.adam(learning_rate=opts.lr, b1=0.9, b2=0.999)
      )
  elif opts.optimizer == 'amsgrad':
    optimizer = optax.chain(
      optax.clip_by_global_norm(opts.grad_clip_val),
      optax.amsgrad(learning_rate=opts.lr, b1=0.9, b2=0.999)
      )
  elif opts.optimizer == 'sgd':
    optimizer = optax.sgd(learning_rate=opts.lr)
  elif opts.optimizer == 'warmup_decay':
    linear_warmup = optax.polynomial_schedule(
      init_value=1e-5,
      end_value=opts.lr,
      power=1.0,
      transition_steps=opts.warmup_steps)
    sqrt_decay = optax.polynomial_schedule(
      init_value=opts.lr,
      end_value=1e-5,
      power=0.5,
      transition_steps=opts.decay_steps)
    lr_schedule = optax.join_schedules(
      schedules=[linear_warmup, sqrt_decay],
      boundaries=[opts.warmup_steps])
    optimizer = optax.chain(
      optax.clip_by_global_norm(5.0),
      optax.scale_by_adam(),
      optax.scale_by_schedule(lr_schedule),)
  elif opts.optimizer == 'cosine':
    lr_schedule = optax.warmup_cosine_decay_schedule(init_value=1e-6,
                                                     peak_value=opts.lr,
                                                     warmup_steps=opts.warmup_steps,
                                                     decay_steps=opts.decay_steps)
    optimizer = optax.chain(
      optax.scale_by_adam(),
      optax.scale_by_schedule(lr_schedule),)
  else:
    raise NotImplementedError
  return optimizer
