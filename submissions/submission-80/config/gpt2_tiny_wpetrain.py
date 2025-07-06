{
    'model_type': 'gpt2',
    'from_config' : True,
    'hidden_size' : 768,
    'intermediate_size' : 3072,
    'num_hidden_layers' : 1,
    'num_attention_heads' : 1,
    'max_position_embeddings' : 2000,
    'vocab_size' : 200,
    # 'onehot_embed': True, # input is onehot, with trainable wpe.
    'onehot_embed': False, # for now, we assume token embedding is not one-hot.
    'wpe_train': True, 
}
