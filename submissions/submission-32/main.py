'''
This file contains code for training models. It relies on main_utils.py
for argparse options. It constructs training and evaluation data iterators,
creates a model (supporting loading from checkpoint logic), and trains the
model. Evaluations are run throughout training and checkpoints are saved.

It also factors in any optogenetic clamps that may be used throughout training.
See artificial_optogenetics_guide.md and opto.py for more details.
'''

from typing import Tuple
from jax import Array
from functools import partial
import os
import shutil
import json
from datetime import datetime
from tqdm import tqdm

import numpy as np
import h5py as h5

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import jax
import jax.numpy as jnp
import equinox as eqx
import optax


import samplers
import main_utils
import opto_ as opto
from metrics import ce, accuracy, entropy, p_correct


ALL_TRAIN_METRICS = ['loss', 'grad_norm', 'grad_batch_stddev']


def smart_index(arr_maybe_none, i, default):
	if arr_maybe_none is None:
		return default
	elif isinstance(arr_maybe_none, str):
		return arr_maybe_none
	else:
		return arr_maybe_none[i]



# Uncomment this jit if looking to microbatch
# @eqx.filter_jit
# Remember that this only yields the grad w.r.t. first positional arg!
@eqx.filter_value_and_grad
def compute_loss(
	model: eqx.Module,
	fwd_fn,
	loss_fn,
	weight_decay: float,
	x: Array, 
	y: Array, 
	keys: Array
) -> Array:
	# pred_y = jax.vmap(model.call_with_all_aux)(x, y, key=keys)['out']
	pred_y = jax.vmap(partial(fwd_fn, model=model))(x=x, y=y, key=keys)['out']
	query_loss = loss_fn(pred_y[:, -1, :], y[:, -1, :])
	# Hacky, but prevents nan'ing gradients of 0 (which biases are initialized too)
	weight_norm = jnp.sum(jnp.stack(jax.tree_map(lambda x: jnp.linalg.norm(jnp.where(x != 0, x, 0)), jax.tree_leaves(eqx.filter(model, eqx.is_array)))))
	return query_loss.mean() + weight_decay * weight_norm


# Comment this jit if looking to microbatch
@eqx.filter_jit
def train_step(
	model: eqx.Module,
	fwd_fn,
	loss_fn,
	optimizer: optax.GradientTransformation,
	opt_state: Array,
	microbs: int,
	weight_decay: float,
	x: Array,
	y: Array,
	key: Array,
) -> Tuple[Array, eqx.Module, Array]:
	keys = jax.random.split(key, x.shape[0])

	losses = []
	grads = []
	for i in range(0, x.shape[0], microbs):
		use_x = x[i:i+microbs]
		use_y = y[i:i+microbs]
		use_keys = keys[i:i+microbs]
		# For some reason using keywords for the argument on the next line breaks things
		# I think it has to do with how equinox implements the value_and_grad wrapper
		# Namely, it "priveleges" 'x' I think
		mini_loss, mini_grad = compute_loss(model, fwd_fn, loss_fn, weight_decay, use_x, use_y, use_keys)
		losses.append(mini_loss)
		grads.append(mini_grad)

	loss = jnp.mean(jnp.array(losses))
	avg_fn = lambda *args: jnp.mean(jnp.array(args),axis=0)
	norm_fn = lambda model_like: jnp.sqrt(jnp.sum(jnp.array([jnp.sum(arr) for arr in jax.tree_util.tree_flatten(model_like)[0]])))

	grad = jax.tree_map(avg_fn, *grads)
	grad_norm = norm_fn(jax.tree_map(lambda x: x**2, grad))
	grads_dev = [jax.tree_map(lambda x,y: (x-y)**2, g, grad) for g in grads]
	grad_batch_var = jax.tree_map(avg_fn, *grads_dev)
	grad_batch_stddev = norm_fn(grad_batch_var)

	update, opt_state = optimizer.update(grad, opt_state)
	model = eqx.apply_updates(model, update)
	# Make sure that ALL_TRAIN_METRICS is correspondingly updated
	# if more metrics are added to this function
	return {'loss': loss, 'grad_norm': grad_norm, 'grad_batch_stddev': grad_batch_stddev}, model, opt_state


@eqx.filter_jit
def eval_step(
	model: eqx.Module, 
	fwd_fn,
	loss_fn,
	x: Array, 
	y: Array, 
	key: Array,
	classification: bool=False,
) -> Tuple[Array, Array]:

	keys = jax.random.split(key, x.shape[0])
	pred_y = jax.vmap(partial(fwd_fn, model=model))(x=x, y=y, key=keys)['out']

	# We don't care about non-final sequence outputs from hereon out
	y = y[:, -1, :]
	if len(pred_y.shape)>2:
		pred_y = pred_y[:, -1, :]
	
	retval = dict()
	retval['loss'] = loss_fn(pred_y, y)
	if classification:
		retval['accs'] = accuracy(pred_y, y)
		retval['entr'] = entropy(pred_y)
		retval['p_corr'] = p_correct(pred_y, y)
		
	return retval


@eqx.filter_jit
def eval_step_gd(
	loss_fn,
	x: Array, 
	y: Array,
	eta: float=5.0,
	classification: bool=False,
):
	def loss(w_, x_, y_):
		y_pred = jnp.einsum('sd,dc->sc', x_[:-1, :], w_)
		return jnp.mean(loss_fn(y_pred, y_[:-1, :]))
	w0 = jnp.zeros((x.shape[0], x.shape[2], y.shape[-1]))
	w1 = w0 - eta * jax.vmap(jax.grad(loss, argnums=0), in_axes=(0, 0, 0))(w0, x, y)
	assert w1.shape==w0.shape, "grad is wrong"
	
	pred_y = jnp.einsum('bd,bdc->bc', x[:, -1, :], w1)
	y = y[:, -1, :]
	
	retval = dict()
	retval['loss'] = loss_fn(pred_y, y)
	if classification:
		retval['accs'] = accuracy(pred_y, y)
		retval['entr'] = entropy(pred_y)
		retval['p_corr'] = p_correct(pred_y, y)
		
	return retval


def make_batched_fn(fn, batch_size):
	def batched_fn(model, x, y, key):
		total = x.shape[0]
		metrics = dict()
		seed = key
		for it in range(0, total, batch_size):
			seed, use = jax.random.split(seed)
			cap = min(it+batch_size, total)
			out = fn(model=model, x=x[it:cap], y=y[it:cap], key=use)
			for m in out:
				metrics.setdefault(m, []).append(out[m])
		for m in metrics:
			metrics[m] = jnp.concatenate(metrics[m])
		return metrics
	return batched_fn


def evaluate(
	model: eqx.Module,
	fwd_fn,
	loss_fn,
	key: Array,
	eval_data,
	eval_batch_size,
	classification=False,
):
	seeds = jax.random.split(key, len(eval_data))
	eval_fn = make_batched_fn(partial(eval_step, fwd_fn=fwd_fn, loss_fn=loss_fn, classification=classification), eval_batch_size)
	retval = dict()
	for seed, k in zip(seeds, eval_data):
		retval[k] = eval_fn(model, eval_data[k]['examples'], eval_data[k]['labels'], key=seed)
	return retval

def run_with_opts(opts):

	### Process input opts ###
	opts.train_microbs = opts.train_bs

	main_utils.check_opts(opts)

	if opts.disable_jit:
		print("Disabling jit")
		jax.config.update('jax_disable_jit', True)

	opts.ckpt_sched = np.arange(0, opts.train_iters, opts.ckpt_every)
	opts.ckpt_every = None

	# Sort and canonicalize for json
	opts.ckpt_sched = [int(x) for x in sorted(opts.ckpt_sched)]

# It's important to set all the defaults before saving the opts
# For example, when loading a checkpoint, we want things like model output classes
# to be filled in

	if opts.raw_name:
		run_name = opts.run
	else:
		run_name = '_'.join([datetime.now().strftime("%Y%m%d%H%M%S"), opts.run])

	run_folder = '/'.join([opts.base_folder, run_name])
	if os.path.exists(run_folder):
		shutil.rmtree(run_folder)
	os.makedirs(run_folder, exist_ok=True)
	with open('/'.join([run_folder, 'config.json']), 'w') as f:
		json.dump(vars(opts), f, indent='\t')

	opts.train_seed = jax.random.PRNGKey(opts.train_seed)
	opts.eval_seed = jax.random.PRNGKey(opts.eval_seed)

	#if opts.ckpt_sched is not None:
	ckpt_folder = '/'.join([run_folder, 'checkpoints'])
	os.makedirs(ckpt_folder, exist_ok=True)

	### Make train and eval data samplers ###
	train_data_seed, train_model_seed = jax.random.split(opts.train_seed, 2)
	_,_, eval_model_seed = jax.random.split(opts.eval_seed, 3)

	if opts.classification:
		train_data_sampler = jax.jit(samplers.make_balanced_classification_queries_sampler(opts.n_labels, opts.train_context_len, opts.train_bs, opts.X_dim, opts.train_noise_scale))


	### Setup model and optimizer ###
	# Includes logic for loading from checkpoints
	fwd_fn_to_use = opto.make_fn_from_opts(opts)
	loss_fn_to_use = ce
	
	if opts.load_from_ckpt is not None:
		assert opts.load_from_ckpt_cfg is not None, "Must specify config for loading model checkpoint"
		print("Resetting config to checkpointed config")
		load_opts = main_utils.get_opts_from_json_file(opts.load_from_ckpt_cfg)
		model = main_utils.get_model_from_opts(load_opts)
	else:
		model = main_utils.get_model_from_opts(opts)

	optimizer = main_utils.get_optimizer_from_opts(opts)
	opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

	start_iter = 0
	if opts.load_from_ckpt is not None:
		ckpt_fmt = {'iter': -1, 
					'seeds': {'eval_model_seed': eval_model_seed,
										'train_data_seed': train_data_seed,
										'train_model_seed': train_model_seed}, 
					'opt_state': opt_state,
					'model': model}
		ckpt = eqx.tree_deserialise_leaves(opts.load_from_ckpt, ckpt_fmt)

		start_iter = ckpt['iter']
		train_data_seed = ckpt['seeds']['train_data_seed']
		train_model_seed = ckpt['seeds']['train_model_seed']
		opt_state = ckpt['opt_state']
		model = ckpt['model']

	### Setup log.h5 ###
	ckpt_ind = np.searchsorted(opts.ckpt_sched, start_iter)

	train_metrics = {m: [] for m in ALL_TRAIN_METRICS}
	# We'll always want train iters
	train_metrics['iter'] = []

	results = h5.File('/'.join([run_folder, 'log.h5']), 'a')
	for m in train_metrics:
		if 'train_{}'.format(m) not in results:
			results.create_dataset('train_{}'.format(m), shape=(0,), maxshape=(None,), dtype=float)
	results.close()

	### MAIN TRAIN LOOP ###
	# Runs eval and checkpoints according to schedule
	# Note i = iterations = # sequences seen

	for i in tqdm(range(start_iter, opts.train_iters, opts.train_bs)):
		if ckpt_ind < len(opts.ckpt_sched) and i >= opts.ckpt_sched[ckpt_ind]:
			#print("Checkpointing...", i)
			ckpt = {'iter': i,
					'seeds': {'eval_model_seed': eval_model_seed,
										'train_data_seed': train_data_seed,
										'train_model_seed': train_model_seed}, 
					'opt_state': opt_state,
					'model': model}
			num = '{}'.format(i).zfill(13)
			eqx.tree_serialise_leaves('/'.join([ckpt_folder, num+".eqx"]), ckpt)
			#if len(train_metrics['iter']) > 0:
				#print(train_metrics['iter'][-1], train_metrics['loss'][-1])
			ckpt_ind += 1

		# Train step -- the train_data_seed is split and passed along (a la functional
		# programming)
		train_data_seed, current_data_seed = jax.random.split(train_data_seed)
		# batch is a dict with keys 'examples' and 'labels'
		# 'examples' is [batch_size, train_context_len + 1, d]
		# 'labels' is [batch_size, train_context_len + 1, C]
		# (the +1 is for the query)
		batch = train_data_sampler(current_data_seed)

		train_model_seed, current_model_seed = jax.random.split(train_model_seed)
		metrics, model, opt_state = train_step(model=model, 
												fwd_fn=fwd_fn_to_use,
												loss_fn=loss_fn_to_use,
												optimizer=optimizer, 
												opt_state=opt_state, 
												microbs=opts.train_microbs, 
												weight_decay=opts.weight_decay,
												x=batch['examples'], 
												y=batch['labels'], 
												key=current_model_seed)
		train_metrics['iter'].append(i+opts.train_bs)
		for m in metrics:
			train_metrics[m].append(metrics[m])

	print("End of training")

	if opts.ckpt_sched[-1] <= opts.train_iters:
		#print("Checkpointing...", i + opts.train_bs)
		ckpt = {'iter': i + opts.train_bs,
				'seeds': {'eval_model_seed': eval_model_seed,
									'train_data_seed': train_data_seed,
									'train_model_seed': train_model_seed}, 
				'opt_state': opt_state,
				'model': model}
		num = '{}'.format(i + opts.train_bs).zfill(13)
		eqx.tree_serialise_leaves('/'.join([ckpt_folder, num+".eqx"]), ckpt)