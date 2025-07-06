from .model import get_model
import torch
import os


def my_get_model(args):
    model_args = eval(open(args.model_config_path).read())
    model_args["num_hidden_layers"] = args.num_hidden_layers
    model_args["num_attention_heads"] = args.num_attention_heads
    if args.hidden_size != None:
        model_args["hidden_size"]=args.hidden_size
        model_args["intermediate_size"]=args.hidden_size*4
    model_args["vocab_size"] = args.vocab_size
    model_args["onehot_embed"] = args.onehot_embed
    model_args["position_embedding"] = args.position_embedding
    print(model_args)
    model = get_model(
        **model_args
    )
    model = model.to(dtype = torch.float32)

    if(args.model_dir):
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'pytorch_model.bin')), strict=True)
        # state_dict = torch.load(os.path.join(args.model_dir, 'pytorch_model.bin'))
        # breakpoint()
        
    return model