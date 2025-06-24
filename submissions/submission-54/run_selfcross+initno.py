import os
import torch
import json
from pipelines.pipeline_sc_initno import SelfCrossInitNOPipeline
import numpy as np
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="SD15")
parser.add_argument("--dataset", type=str, default="animals")
parser.add_argument("--num_initial_noises", type=int, default=200)
parser.add_argument("--updates_per_noise", type=int, default=10)
args = parser.parse_args()

model_choice = args.model
dataset = args.dataset
num_initial_noises = args.num_initial_noises
bon_optim = args.updates_per_noise
total_function_call = num_initial_noises * bon_optim


# ---------
# Arguments
# ---------
# model_choice    = "SD15" #, "SD15"
SD15_VERSION    = "runwayml/stable-diffusion-v1-5"
# SD21_VERSION    = "stabilityai/stable-diffusion-2-1"
token_indices   = []

# dataset         = "animals"
# dataset         = "animals_objects"
# dataset         = "objects"
# dataset         = "SSD-2"
# dataset         = ["red apple and blue ball"]

# total_function_call = 300
# bon_optim = 10

run_sd          = False

def Convert(string):
    li = list(string.split(" "))
    return li


def main():
    if model_choice == "SD15":
        pipe = SelfCrossInitNOPipeline.from_pretrained(SD15_VERSION).to("cuda:0")
        attn_res = (16, 16)
        max_iter_to_alter = 25
        run_initno = True
    else:
        raise Exception("model_choice should be SD15.")

    if run_sd ==True: run_initno = False

    if dataset in ["animals", "animals_objects", "objects", "SSD-2"]:
        with open('prompts.txt') as f:
            data = f.read()
        print("Data type before reconstruction : ", type(data))
        # reconstructing the data as a dictionary
        prompts = json.loads(data)
        print("seeds Data type after reconstruction : ", type(prompts))
        print(prompts)
        prompts = prompts[dataset]
    else:
        prompts = [dataset]

    with open('seeds.txt') as f:
        data = f.read()
    print("seeds Data type before reconstruction : ", type(data))
    # reconstructing the data as a dictionary
    seeds = json.loads(data)
    print("seeds Data type after reconstruction : ", type(seeds))
    print(seeds)

    for PROMPT in prompts:
        # use get_indices function to find out indices of the tokens you want to alter
        words = Convert(PROMPT)
        if len(words) == 2:
            token_indices = [2]
        elif len(words) == 5:
            token_indices = [2, 5]
        elif len(words) == 6:
            token_indices = [2, 6]
        elif len(words) == 7:
            token_indices = [3, 7]
        else:
            raise Exception("for '{}' token_indices cannot be specified automatically.".format(words))
        print(PROMPT, token_indices)

        pipe.get_indices(PROMPT)
        path = './outputs/initno_selfcross_noise_{}_update_{}/{}/{}'.format(num_initial_noises, bon_optim, dataset, PROMPT)
        os.makedirs('{:s}'.format(path), exist_ok=True)
        for SEED in seeds:
            SEED=int(SEED)
            print('Seed ({}) Processing the ({}) prompt'.format(SEED, PROMPT))
            generator = torch.Generator("cuda").manual_seed(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            random.seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            np.random.seed(SEED)
            images = pipe(prompt=PROMPT,token_indices=token_indices,guidance_scale=7.5,generator=generator,
                          num_inference_steps=50,max_iter_to_alter=max_iter_to_alter, attn_res=attn_res,
                          K=16,seed=SEED,run_sd=run_sd,run_initno=run_initno, total_function_call=total_function_call, bon_optim=bon_optim).images
            images[0].save(path +f"/{SEED}.png")
            
            print(f"\n\nSeed {SEED} completed: {PROMPT}\n\n")


if __name__ == '__main__':
    main()

