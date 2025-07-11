#load packages
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import math, time
import sys
sys.path.append('../..')
import os
from functions.train_test import train, test
from functions.get_data import data_n_loaders
from models import SAE
import wandb
import argparse
from datetime import date
from functions.utils import load_args_from_file, softplus_inverse, resample_deadlatents
from torchsummary import summary
import csv
import argparse
import time
from functions.utils import read_hyperparameters

if __name__=='__main__':

    #read parameters from hyperparameters file in array job (appropriate line of hyperparameters2.csv)
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--task_id', type=int, required=True, 
                            help='SLURM task ID')
    argsX = parser.parse_args()
    hyperparameters = read_hyperparameters(argsX.task_id - 1, './hyperparameters2.csv')
    
    #save hyperparameters- SAE type, K in TopK, and scaling of regularizer loss (gamma_reg)
    if hyperparameters is not None:
        sae_type = hyperparameters['sae_type']
        kval_topk = int(hyperparameters['kval_topk'])
        gamma_reg = float(hyperparameters['gamma_reg'])
        # scale = hyperparameters['scale']
    else:
        raise ValueError(f"No sae hyperparams found for task {argsX.task_id}")
    
   
    
    #create necessary subfolders for status files, models, figures
    ARGS_FROM_FILE = True
    RESULTS_PATH = './'
    FIGURES_PATH = RESULTS_PATH+'figs/'
    # LAB_DIR = os.environ['USERDIR'] 
    # DATA_PATH = LAB_DIR+'/data'
    STATUS_PATH = RESULTS_PATH + 'status_files/'
    SAVE_MODELS_PATH = RESULTS_PATH+'saved_models/'

    #create directories if they don't exist
    if not os.path.exists(FIGURES_PATH):
        os.makedirs(FIGURES_PATH)
    if not os.path.exists(STATUS_PATH):
        os.makedirs(STATUS_PATH)
    if not os.path.exists(SAVE_MODELS_PATH):
        os.makedirs(SAVE_MODELS_PATH)

    #get parameters from settings file- 
    # note that parameters loaded from hyperparameters file overwrite any parameters in settings file
    args = load_args_from_file('./settings.txt')
    DATA_PATH = args.data_path 

    #from hyperparameters file: overwrites settings file args
    args.sae_type = sae_type 
    args.kval_topk = kval_topk

    #set regularizer based on nonlinearity if 'default' is given in settings
    if args.regularizer == 'default':
        
        #L1 for ReLU
        if args.sae_type=='relu':
            args.regularizer = 'l1'

        #Auxiliary loss for TopK
        elif args.sae_type=='topk':
            # args.regularizer = None
            args.regularizer = 'auxloss'
        elif args.sae_type=='topk_relu':
            # args.regularizer = None
            args.regularizer = 'auxloss'
        
        #L0 for JumpReLU
        elif args.sae_type=='jumprelu':
            args.regularizer = 'l0'
        
        #Distance-weighted L1 for SpaDE
        elif args.sae_type=='sparsemax_dist':
            args.regularizer = 'dist_weighted_l1'
    else:
        args.regularizer = args.regularizer if args.regularizer!='None' else None


    #set gamma_reg based on regularizer if 'default' is given in settings
    if args.gamma_reg=='default':
        if args.regularizer in ['l1', 'dist_weighted_l1', None]:
            args.gamma_reg = 0.1
        elif args.regularizer=='l0':
            args.gamma_reg = 0.01 #l0 loss observed to be larger; smaller gamma to compensate
        elif args.regularizer=='auxloss':
            args.gamma_reg = 1.0
    elif args.gamma_reg=='None':
        args.gamma_reg = 0.0
   
    #decoder weights normalized for ReLU, TopK and JumpReLU SAEs (NOT for SpaDE)
    if args.normalize_decoder=='default':
        if args.sae_type=='relu':
            args.normalize_decoder = True
        elif args.sae_type=='topk':
            args.normalize_decoder = True
        elif args.sae_type=='topk_relu':
            args.normalize_decoder = True
        elif args.sae_type=='jumprelu':
            args.normalize_decoder = True
        elif args.sae_type=='sparsemax_dist':
            args.normalize_decoder = False


    #from hyperparameters file: overwrites settings file args
    args.gamma_reg = gamma_reg
    sae_width = args.sae_width
    device = args.device
    if device=='cuda':
        torch.cuda.empty_cache()

    
    #training params
    MOMENTUM = 0.9
    EPOCHS = 1 #for multiepoch training over entire data (we train online so this is not used)
    LEARNING_RATE = 1e-2
    WEIGHT_DECAY = 0.0

    #experiment name using random words
    from wonderwords import RandomWord
    import random
    seedx = random.randint(0, 1000)
    random.seed(seedx)
    r = RandomWord()
    word = r.word(word_min_length=2, word_max_length=5)
    date = date.today().strftime("%m%d%y") if args.experiment_date=='today' else args.experiment_date

    # define experiment name using relevant parameters
    kvalue_str = "k"+str(args.kval_topk)+"_" if 'topk' in args.sae_type else ''
    gamma_str = "gamreg"+str(args.gamma_reg)+"_" if args.sae_type!='topk_relu' else ''
    # hyperparams_term = f"lr{LEARNING_RATE}_noWD_" if WEIGHT_DECAY==0.0 else ""
    saename = args.sae_type if args.sae_type!='sparsemax_dist' else 'spade'
    widthterm = 'w'+str(sae_width)+'_' if sae_width!=50 else ''
    EXPT_NAME = word+str(seedx)+"_"+saename +"_" +\
        gamma_str+kvalue_str + widthterm + date

    #status file- track training progress
    STATUS_FILE = STATUS_PATH+'status_'+EXPT_NAME+'.txt'
    def update_status(text, option="a+"):
        with open(STATUS_FILE, option) as f:
            f.write('\n'+text)
    update_status(f"Using {device} device",option='w') #mention device
    for arg, value in vars(args).items(): #write experiment settings into status file
        update_status(f"{arg}: {value}")

    #load data (preprocessed)
    train_dataloader, test_dataloader,\
        train_data, test_data = data_n_loaders(args.dataset, args.batch_size, \
                                                return_data=True, data_path=DATA_PATH,\
                                                    standardise_data = True)

    #define model 
    model = SAE(dimin=args.data_dim, width=sae_width, sae_type=args.sae_type, \
        kval_topk=args.kval_topk, normalize_decoder=args.normalize_decoder)
    model = model.to(device)


    #init weights with data samples
    if args.weight_init=='data': #data init
        torch.manual_seed(args.seed_id)
        num_train_ex = len(train_data)
        indices_examples = torch.randperm(num_train_ex)
        indices_examples = indices_examples[:sae_width] #choose sae_width indices
        enc_init = torch.zeros((sae_width, args.data_dim)).to(device)
        with torch.no_grad():
            for k in range(sae_width):
                enc_init[k,:] = train_data[indices_examples[k]][0].squeeze().to(device)
            model.Ae.copy_(enc_init)
            model.Ad.copy_(enc_init.T)

    #optimization parameters
    if args.optimizer=="sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    elif args.optimizer=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, \
                                    weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError("optimizer must be one of 'sgd', 'adam'")
        

    #sync progress with wandb- weights and biases
    wandb.init(
        # set the wandb project where this run will be logged
        project=args.wandbprojectname,
        name=EXPT_NAME,
        # track hyperparameters and run metadata
        config=vars(args)
    )

    #number of iterations of minibatches for online training
    ITERSnum = math.ceil(len(train_data)/args.batch_size) if args.online_training else EPOCHS

    #set learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ITERSnum, eta_min=1e-4) 

   #initialize values to track over training
    train_loss_epoch = [None for i in range(ITERSnum)]
    test_loss_epoch = [None for i in range(ITERSnum)]
    lambda_vals = [None for i in range(ITERSnum)]
    
    #save checkpoints
    if args.save_checkpoints:
        checkpoint_loc = SAVE_MODELS_PATH+'checkpoints_'+EXPT_NAME+'/'
        if not os.path.exists(checkpoint_loc):
            os.makedirs(checkpoint_loc)
        
        #first save a copy of the args settings file for this experiment
        import shutil
        shutil.copy('./settings.txt', checkpoint_loc+'settings.txt')
        
        #save checkpoint at init
        savecount = 0
        fname_chk = lambda id: checkpoint_loc+ 'model_'+str(id)+'epochs.pt'
        torch.save({'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'lr_scheduler':scheduler.state_dict()},fname_chk(savecount))
        savecount+=1
        
        #to resample dead neurons- not used
        if args.resample_deadneurons:
            track_epochs = [2**i-1 for i in range(math.floor(math.log2(ITERSnum))+1)] + \
                [ITERSnum-1] + [int(EPOCHS/2)+i for i in range(-1, 6)]
            track_epochs = sorted(list(set(track_epochs)))
        else:
            track_epochs = [2**i-1 for i in range(math.floor(math.log2(ITERSnum))+1)] + [ITERSnum-1]
    torch.manual_seed(0)
    tic = time.perf_counter()


    # for multi-epoch training
    if not args.online_training:
        for t in range(EPOCHS):
            update_status(f"Epoch {t+1}\n-------------------------------")

            # train model and get test loss
            train_loss_epoch[t] = train(train_dataloader, model, optimizer, \
                                            update_status_fn=update_status, regularizer=args.regularizer, \
                                                encoder_reg=args.encoder_reg, gamma_reg=args.gamma_reg, \
                                                    return_concept_loss=args.return_concept_loss, \
                                                        num_concepts=args.num_concepts, clip_grad=args.clip_grad)
                
            test_loss_epoch[t] = test(test_dataloader, model, update_status_fn=update_status,\
                                        regularizer=args.regularizer, encoder_reg=args.encoder_reg, gamma_reg=args.gamma_reg, \
                                            return_concept_loss=args.return_concept_loss, num_concepts=args.num_concepts)  
            lambda_vals[t] = model.lambda_val.data
            
            #save model checkpoints
            if args.save_checkpoints:
                if t in track_epochs:
                    torch.save({'model':model.state_dict(),
                            'optimizer':optimizer.state_dict(),
                            'lr_scheduler':scheduler.state_dict()},fname_chk(t+1))
                    savecount+=1
            if args.resample_deadneurons:
                if t==int(EPOCHS/2)-1:
                    with torch.no_grad():
                        resample_deadlatents(model, train_dataloader, num_batches=15)
                    if args.save_checkpoints:
                            torch.save({'model':model.state_dict(),
                                    'optimizer':optimizer.state_dict(),
                                    'lr_scheduler':scheduler.state_dict()}, checkpoint_loc+ 'model_'+str(t+1.5)+'epochs.pt')
                            savecount+=1
            scheduler.step()
            logdata = {"loss_train_mse":train_loss_epoch[t][0], "loss_train_reg":train_loss_epoch[t][1],\
                        "loss_test_mse":test_loss_epoch[t][0], "loss_test_reg":test_loss_epoch[t][1], \
                        "loss_train": train_loss_epoch[t][0]+train_loss_epoch[t][1],\
                        "loss_test": test_loss_epoch[t][0]+test_loss_epoch[t][1], \
                            "lambda": lambda_vals[t]}
            if args.return_concept_loss:
                log_concept_loss = {f"c{i}" + "_loss_train_mse":train_loss_epoch[t][-1][i] for i in range(len(train_loss_epoch[t][-1]))}
                logdata.update(log_concept_loss)
                log_concept_test_loss = {f"c{i}" + "_loss_test_mse":test_loss_epoch[t][-1][i] for i in range(len(test_loss_epoch[t][-1]))}
                logdata.update(log_concept_test_loss)
            wandb.log(logdata)

    #online training, go through entire data only once 
    else: 
        from torch.utils.data import DataLoader, TensorDataset
        g_tr = torch.Generator()
        g_tr.manual_seed(0)
        for t, (batchdata, batchlabels) in enumerate(train_dataloader):
            update_status(f"Epoch {t+1}\n-------------------------------")
            
            #load minibatch and train
            batchdataloader = DataLoader(TensorDataset(batchdata, batchlabels), batch_size=args.batch_size, generator=g_tr)
            train_loss_epoch[t] = train(batchdataloader, model, optimizer, \
                                            update_status_fn=update_status, \
                                                regularizer=args.regularizer, encoder_reg=args.encoder_reg, \
                                                    gamma_reg=args.gamma_reg, \
                                                        return_concept_loss=args.return_concept_loss, \
                                                            num_concepts=args.num_concepts, clip_grad=args.clip_grad)
            lambda_vals[t] = model.lambda_val.data
            scheduler.step()
            logdata = {"loss_train_mse":train_loss_epoch[t][0], "loss_train_reg":train_loss_epoch[t][1],\
                            "loss_train": train_loss_epoch[t][0]+train_loss_epoch[t][1],\
                                "lambda": lambda_vals[t]}
            if args.return_concept_loss:
                log_concept_loss = {f"c{i}" + "_loss_train_mse":train_loss_epoch[t][-1][i] for i in range(len(train_loss_epoch[t][-1]))}
                logdata.update(log_concept_loss)
            wandb.log(logdata)
            
            #save model checkpoints
            if args.save_checkpoints:
                if t in track_epochs:
                    torch.save({'model':model.state_dict(),
                            'optimizer':optimizer.state_dict(),
                            'lr_scheduler':scheduler.state_dict()},fname_chk(t+1))
                    savecount+=1
            if args.resample_deadneurons:
                numbatches = len(train_dataloader)
                if t%(numbatches//3) == (numbatches//3)-1:
                    with torch.no_grad():
                        resample_deadlatents(model, train_dataloader, num_batches=15)
                    if args.save_checkpoints:
                            torch.save({'model':model.state_dict(),
                                    'optimizer':optimizer.state_dict(),
                                    'lr_scheduler':scheduler.state_dict()}, checkpoint_loc+ 'model_'+str(t+1.5)+'epochs.pt')
                            savecount+=1
    toc = time.perf_counter()

    #final status updates
    update_status("Done!")
    update_status(f"Time to train {EPOCHS} epochs = {round(toc-tic,2)}s ({round((toc-tic)/EPOCHS,2)}s per epoch)")

    #save losses
    if args.online_training:
        torch.save({'train_loss':train_loss_epoch, 'lambda':lambda_vals}, checkpoint_loc+'losses.pt')
    else:
        torch.save({'train_loss':train_loss_epoch, 'test_loss':test_loss_epoch, 'lambda':lambda_vals}, checkpoint_loc+'losses.pt')

    wandb.finish()