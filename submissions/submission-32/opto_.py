'''
This file contains argparse options and forward function
constructors for all artificial optogenetics experiments involving
clamping throughout training.

See artificial_optogenetics_guide.md for more information
'''

import argparse

def add_args_to_parser(parser):
  return
  
def get_default_cache_and_mask(depth):
  '''
  Creates a default cache.

  Note we could also use jnp.zeros_like on the first pass of
  the model. Since we may not always do two passes (e.g. when
  just fixing a single head to be an induction head), we found
  this method useful.
  '''
  cache = dict()
  cache_mask = dict()
  cache.setdefault('transformer_output', dict()).setdefault(
                    'block_outputs', 
                    [dict(attn_output=dict()) for i in range(depth)]
                    )
  cache_mask.setdefault('transformer_output', dict()).setdefault(
                        'block_outputs', 
                        [dict(attn_output=dict()) for i in range(depth)]
                        )
  return cache, cache_mask

def add_defaults_if_not_present(opts):
  '''
  Method that adds default arguments for opto to a parser output. Useful
  for restoring from older model configs (that were trained without opto 
  or with older opto arguments)
  '''
  parser = argparse.ArgumentParser()
  add_args_to_parser(parser)
  defaults = vars(parser.parse_args([]))
  to_modify = vars(opts)
  for k in defaults:
    if k not in to_modify:
      to_modify[k] = defaults[k]

def default_model_fwd_fn(model, x, y, key):
  '''
  Default forward function, used to match signature that is returned by make_fn_from_opts
  '''
  return model.call_with_all_aux(examples=x, labels=y, key=key, cache=dict(), cache_mask=dict())

def make_fn_from_opts(opts, default_fn=default_model_fwd_fn):
  '''
  The opts will be used as specified above

  Returns:
    A callable function that takes in model, single x, single y, single key
    and return call_with_all_aux output with the specified caching scheme.

    This forward function can then be vmapped to process batches of data.

    If no opto parameters are specified, this returns the dummy default_model_fwd_fn
  '''
  
  add_defaults_if_not_present(opts)
  return default_fn