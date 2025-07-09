from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerControl, TrainerState, TrainerCallback
import transformers
transformers_version = '.'.join(transformers.__version__.split('.')[:2])
if float(transformers_version) >= 4.16:
    # "evaluation_strategy" has been deprecated. 
    NEW_TRANSFORMERS = True
else:
    NEW_TRANSFORMERS = False

import wandb
import torch
from .utils import set_seed, parse_args
from .utils import encoder, decoder
from .my_get_model import my_get_model
from .utils_training import padding, CustomDataCollator, CustomDataset
from .utils_training import get_graph_path, get_output_dir, get_task_generator
import os
import copy


def generate_and_check_valid(encoder, decoder, model, input_ids, attention_mask, TaskGenerator, input_ids_before_padding, max_length=300, eos_token_id=6, pad_token_id=0):
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id 
        )
    model.train()

    generated_tokens = generated_ids[:, input_ids.size(1):] 

    valid_count = 0
    for i in range(generated_tokens.size(0)):
        path = [decoder(token.item()) for token in generated_tokens[i] if token != pad_token_id]
        if TaskGenerator.check_valid(S = decoder(input_ids_before_padding[i][0]), T = decoder(input_ids_before_padding[i][1]), path=path):
            valid_count += 1

    return valid_count

def compute_metric_from_data(model, input_ids, TaskGenerator, encoder, decoder, eval_size=512, max_length=300, batch_size=2048):
    valid_count = 0
    for i in range(0, eval_size, batch_size):
        batch_input_ids = input_ids[i:min(i + batch_size, eval_size)] 
        batch_input_ids_before_padding = copy.deepcopy(batch_input_ids)
        batch_input_ids, batch_attention_mask = padding(batch_input_ids, padding_direction="left")
        batch_input_ids = torch.tensor(batch_input_ids).to(model.device)  
        batch_attention_mask = torch.tensor(batch_attention_mask).to(model.device)

        valid_count_batch = generate_and_check_valid(encoder=encoder, decoder=decoder, model=model, input_ids=batch_input_ids, attention_mask=batch_attention_mask, TaskGenerator=TaskGenerator, max_length=max_length, input_ids_before_padding = batch_input_ids_before_padding)

        valid_count += valid_count_batch
    valid_percentage = valid_count / eval_size
    return valid_percentage

def compute_metrics(model, encoder, decoder, train_dataset, val_dataset, TaskGenerator, no_eval = False, eval_size = 512, batch_size=2048, max_length=300): 
    result = {}
    if no_eval == False:
        data_iter = iter(val_dataset)
        eval_input_ids = []
        for i in range(eval_size):
            data = next(data_iter)
            input_ids = data["input_ids"]
            if TaskGenerator.task_type == "SFT9" and input_ids[3] == encoder("Type0"):
                input_ids[3] = encoder("Type1")
            
            labels = data["labels"]
            filtered_input_ids = [input_ids[idx] for idx, label in enumerate(labels) if label == -100]
            eval_input_ids.append(filtered_input_ids)

        eval_valid_percentage = compute_metric_from_data(model, eval_input_ids, TaskGenerator, encoder, decoder, eval_size=eval_size, max_length=max_length, batch_size=batch_size)
        result["eval_valid_percentage"] = eval_valid_percentage
    
    data_iter = iter(train_dataset)
    train_input_ids = []
    for i in range(eval_size):
        data = next(data_iter)
        input_ids = data["input_ids"]
        if TaskGenerator.task_type == "SFT9" and input_ids[3] == encoder("Type0"):
            input_ids = data["input_ids"]
        labels = data["labels"]
        filtered_input_ids = [input_ids[idx] for idx, label in enumerate(labels) if label == -100]
        train_input_ids.append(filtered_input_ids)
    # breakpoint()

    train_valid_percentage = compute_metric_from_data(model, train_input_ids, TaskGenerator, encoder, decoder, eval_size=eval_size, max_length=max_length, batch_size=batch_size)
    result["train_valid_percentage"] = train_valid_percentage

    return result

class CustomTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None):
        if self.compute_metrics is not None:
            metrics = self.compute_metrics()
        else:
            metrics = {}
        self.log(metrics)
        return metrics

def train(args):
    print("args: ", args)
    set_seed(args.random_seed)

    if args.model_dir != None and args.model_dir != "None":
        
        if "checkpoint" in args.model_dir or "tmp" in args.model_dir:
            base_model_name = '@' + args.model_dir.split('/')[-2].split('@')[-1]
        else:
            base_model_name = '@' +  args.model_dir.split('/')[-1].split('@')[-1]
    else:
        args.model_dir = None
        base_model_name = None

    graph_path=get_graph_path(args)
    output_dir = get_output_dir(args, graph_path=graph_path, base_model_name=base_model_name)

    wandb.init(project="Anchoring", config=args,
               name = output_dir.split('/')[-1])
    try:
        graph_path = os.path.join(args.graph_dir, f"{graph_path}.json")
        TaskGenerator = get_task_generator(args, graph_path)
        train_dataset = CustomDataset(TaskGenerator, "train")
        eval_dataset = CustomDataset(TaskGenerator, "eval") # from the same generator, so that the evaluation is also fresh. 

        model = my_get_model(args)

        data_collator = CustomDataCollator(pad_token_id=0) 
        if NEW_TRANSFORMERS:
            SFT_args = TrainingArguments(
                output_dir=output_dir,          
                max_steps=args.max_steps,
                per_device_train_batch_size=args.batch_size,  
                warmup_steps=0,                
                weight_decay=args.weight_decay,               
                logging_dir='./logs',            
                logging_steps= args.log_steps,
                save_steps = args.save_steps,
                save_total_limit = 1,
                eval_strategy="steps",     
                eval_steps= args.eval_steps, 
                learning_rate = args.lr,
                label_names = ['labels'],
                save_safetensors = False, 
                report_to="wandb",
                lr_scheduler_type="linear",
            )
        else:
            SFT_args = TrainingArguments(
                output_dir=output_dir,          
                max_steps=args.max_steps,
                per_device_train_batch_size=args.batch_size,  
                warmup_steps=args.warmup_ratio * args.max_steps,                
                weight_decay=args.weight_decay,             
                logging_dir='./logs',            
                logging_steps= args.log_steps,
                save_steps = args.save_steps,
                save_total_limit = 1,
                evaluation_strategy="steps",     
                eval_steps= args.eval_steps, 
                learning_rate = args.lr,
                label_names = ['labels'],
                save_safetensors = False, 
                report_to="wandb",
                lr_scheduler_type="linear",
            )

        eval_size=args.eval_size
        no_eval = (args.train_type in ["pretrain2", "TwoTrees_pt1", "TwoTrees_pt2", "TwoTrees_pt3", "Two-levelDumbbell_pt2", "Two-levelDumbbell_pt1", "Two-levelGraph_pt1", "KPartiteGraph_pt1", "Two-levelGraph_pt2"] or args.eval_rate == 0)

        def compute_metrics_with_val_dataset(): # Cleverly pa[ssed as an additional parameter.
            return compute_metrics(model=model, encoder=encoder, decoder = decoder,
                                train_dataset=train_dataset, val_dataset=eval_dataset, eval_size=eval_size,
                                TaskGenerator = TaskGenerator, no_eval=no_eval, max_length=args.max_generation_length)
        
        

        class WeightNormTrackerCallback(TrainerCallback):
            """
            Log weight norms.
            """

            def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
                """
                This is called at the end of every training step. 
                You can also use on_epoch_end if you only want it once per epoch.
                """

                # Currently hardcoded to log every 256 steps at the beginning / 2048 later.
                # if state.global_step % 2048 == 0:
                if 1:
                    model = kwargs["model"]
                    
                    norms = {}
                    total_norm = 0.0 # track the total norm of all params
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            param_norm = param.data.norm(2)  # L2 norm of the parameter
                            norms[name] = param_norm.item()
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    norms["total_norm"] = total_norm

                    wandb.log(norms, step=state.global_step)


        eval_dataset = eval_dataset if NEW_TRANSFORMERS else None
        SFT_trainer = CustomTrainer(
            model=model, args=SFT_args, 
            train_dataset=train_dataset,
            eval_dataset=eval_dataset, 
            data_collator = data_collator,
            compute_metrics=compute_metrics_with_val_dataset,
            callbacks=[WeightNormTrackerCallback()],
        )

        SFT_trainer.train()
        SFT_trainer.save_model(output_dir=output_dir)
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()