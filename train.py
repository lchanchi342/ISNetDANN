#train.py


import argparse
import collections
import json
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
import torch.utils.data
import pickle
from pathlib import Path
import math

from torch.utils.data import DataLoader

import hparams_registry
from model_training import misc, datasets, algorithms, early_stopping
from model_training.dataloaders import FastDataLoader
from model_eval import eval_helper

from argparse import ArgumentParser
import torch.utils.data as Tdata
from argparse import ArgumentParser
import torch.utils.data as Tdata

#imports isnet
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
import torchvision as tv
import torch.utils.data as data
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset
import copy
import matplotlib
import warnings



if __name__ == "__main__":
    
    #========================arguments==============================
    parser = argparse.ArgumentParser(description='Shortcut Learning in Chest X-rays')
    # training
    parser.add_argument('--store_name', type=str, default='debug')
    parser.add_argument('--dataset', type=str, default=["MIMIC"], nargs='+')
    parser.add_argument('--task', type=str, default="No Finding", choices=datasets.TASKS + datasets.ATTRS)
    parser.add_argument('--attr', type=str, default="sex", choices=datasets.ATTRS)
    parser.add_argument('--group_def', type=str, default="group", choices=['group', 'label'])
    parser.add_argument('--algorithm', type=str, default="ERM", choices=algorithms.ALGORITHMS)
    # others
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for random hparams (0 for "default hparams")')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    #parser.add_argument('--steps', type=int, default=None)  # for step-controlled training    
    parser.add_argument('--epochs', type=int, default=None) # for epoch-controlled training
    
    # early stopping
    parser.add_argument('--use_es', action='store_true')
    parser.add_argument('--min_delta', type=float, default=0.0, help='Minimum improvement to reset early stopping counter') 
    parser.add_argument('--es_strategy', choices=['metric'], default='metric')
    #parser.add_argument('--es_metric', type=str, default='min_group:accuracy')
    parser.add_argument('--es_metric', type=str, default='overall:AUROC')
    parser.add_argument('--es_patience', type=int, default=5, help='Stop after this many checkpoints w/ no improvement')
    # checkpoints
    parser.add_argument('--resume', '-r', type=str, default='') 
    parser.add_argument('--checkpoint_freq', type=int, default=None, help='Checkpoint every N steps')
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--debug', action='store_true') #keep False to save out.txt y err.txt
    # architectures and pre-training sources
    parser.add_argument('--image_arch', default='densenet121',
                        choices=['densenet121'])
    # data augmentations
    parser.add_argument('--aug', default='basic2',
                        choices=['none', 'basic', 'basic2', 'auto_aug', 'rand_aug', 'trivial_aug', 'augmix'])

    #args ISNET
    parser.add_argument('--heat', action='store_true') #to activate heatmap analysis
    parser.add_argument('--use_masks', action='store_true') #to use masks
    parser.add_argument('--loadCuts', action='store_true')
    parser.add_argument('--tuneCut', action='store_true')
    parser.add_argument('--only_tune_cut', action='store_true')
    
    args = parser.parse_args()

    #==============================================================
    
    start_step = 0
    misc.prepare_folders(args) #create folders
    output_dir = os.path.join(args.output_dir, args.store_name) # outpur_dir default: output/debug

    #save out.txt y err.txt
    if not args.debug:
        sys.stdout = misc.Tee(os.path.join(output_dir, 'out.txt'))
        sys.stderr = misc.Tee(os.path.join(output_dir, 'err.txt'))

    # Execution environment information
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    #print all Args
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    #hiperparameteres config. hparams_seed == 0 for default_parameters.
    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, misc.seed_hash(args.hparams_seed))
    
    #load from json
    if args.hparams:
        hparams.update(json.loads(args.hparams)) 

    # Update hparams with Args info
    hparams.update({
        'image_arch': args.image_arch, #default: DenseNet121
        'data_augmentation': args.aug, #default: basic2/JointTransformBasic2
        'task': args.task, #default: No finding 
        'attr': args.attr, #default: sex
        'group_def': args.group_def, #default: group
        'tuneCut': args.tuneCut #default: False
    })
    

    #Print Hiperparameters
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    #save args
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    

    #----------for reproducibility and determinism--------
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    random.seed(args.seed) # Sets(Fija) the seed for Python's random
    np.random.seed(args.seed) # Sets the seed for NumPy
    torch.manual_seed(args.seed) # Sets the seed in PyTorch, for weight initialization
    
    # CuDNN y PyTorch determinism
    torch.backends.cudnn.deterministic = True # Makes GPU operations deterministic, i.e., always producing the same result given the same data and model
    torch.backends.cudnn.benchmark = False # Disables the search for the fastest algorithms (some of which are non-deterministic)


    torch.use_deterministic_algorithms(True) # Forces deterministic algorithms
   
    # Settings to avoid conflicts with parallelization
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.multiprocessing.set_sharing_strategy('file_system')

    # Defines whether training will run on GPU (cuda) or CPU (cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #info de GPU: 8
    #NVIDIA RTX A6000

    #----------------------Dataset creation------------------
    # Creates the datasets using the corresponding class, e.g., MIMIC(train, ...)
    # for training, testing, and validation
    if len(args.dataset) == 1:
        if args.dataset[0] in vars(datasets):
            train_dataset = vars(datasets)[args.dataset[0]](args.data_dir, 'tr', hparams, group_def=args.group_def, use_masks=args.use_masks)
            val_dataset = vars(datasets)[args.dataset[0]](args.data_dir, 'va', hparams, group_def='group', use_masks=args.use_masks )
            test_dataset = vars(datasets)[args.dataset[0]](args.data_dir, 'te', hparams, group_def='group', use_masks=False) # use_masks is not enabled
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    # variables obtained from the loaded datasets
    num_workers = train_dataset.N_WORKERS
    input_shape = train_dataset.INPUT_SHAPE
    num_labels = train_dataset.num_labels
    #print(f"[DEBUG] train_dataset.num_labels: {train_dataset.num_labels}")

    num_attributes = train_dataset.num_attributes
    data_type = train_dataset.data_type
    #n_steps = args.steps or train_dataset.N_STEPS

    num_epochs = args.epochs

    #updates num_epochs/n_steps
    hparams.update({
        #"steps": n_steps
        "epochs": num_epochs
    })

    print(f"num_attributes: {num_attributes}")
    
    print(f"num_epochs: {num_epochs}")

    #print(f"checkpoint: {num_epochs}")

    #prints datasets info
    print(f"Dataset:\n\t[train]\t{len(train_dataset)}"
          f"\n\t[val]\t{len(val_dataset)}")


    #-------------Create DataLoaders from the previously created datasets----------------

    train_loader = FastDataLoader( # DataLoader wrapper with slightly improved speed by not respawning worker processes at every epoch
        dataset=train_dataset, #train set MIMIC
        batch_size=min(len(train_dataset), hparams['batch_size']), #ensures the batch size is not larger than the training dataset
        num_workers=num_workers #16 
    )

    steps_per_epoch=len(train_loader)
    print(f"steps_per_epoch: {steps_per_epoch}")

    num_steps=steps_per_epoch*num_epochs


    if args.checkpoint_freq is None: #1000, set in experiments
        args.checkpoint_freq = math.ceil(len(train_loader))
            
    checkpoint_freq = args.checkpoint_freq or train_dataset.CHECKPOINT_FREQ


    split_names = ['va', 'te'] #eval splits

    # -----------------------Data loading for evaluation--------------
    eval_loaders = [DataLoader( # Uses a standard DataLoader
        dataset=dset,
        batch_size=max(128, hparams['batch_size'] * 2), # Larger batch size -> 128 for MIMIC
        num_workers=num_workers,
        shuffle=False)
        for dset in [val_dataset, test_dataset]
    ]

    
    # ---------------Algorithm creation, e.g., ERM(....)------------------------

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(data_type, input_shape, num_labels, num_attributes, len(train_dataset), hparams,
                                grp_sizes=train_dataset.group_sizes, attr_sizes=train_dataset.attr_sizes)

    #------------------early stopping config----------------------
    es_group = args.es_metric.split(':')[0]
    es_metric = args.es_metric.split(':')[1]
    es = early_stopping.EarlyStopping(
        patience=args.es_patience, lower_is_better=early_stopping.lower_is_better[es_metric], min_delta=args.min_delta)
    best_model_path = os.path.join(output_dir, 'model.best.pkl')


    # Load a checkpoint to resume training
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            #checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            start_step = checkpoint['step']
            algorithm.load_state_dict(checkpoint['model_dict'])
            #algorithm.optimizer.load_state_dict(checkpoint["optimizer_dict"])
            es = checkpoint['early_stopper']
            print(f"===> Loaded checkpoint (epoch [{start_epoch}])")
            print(f"===> Loaded checkpoint '{args.resume}' (step [{start_step}])")
        else:
            print(f"===> No checkpoint found at '{args.resume}'")

    # Move the model to the GPU
    algorithm.to(device)

    checkpoint_vals = collections.defaultdict(lambda: []) # Dictionary to store metrics throughout training
    
    # Function to save checkpoints
    def save_checkpoint(save_dict, filename='model.pkl'):
        if args.skip_model_save: # to skip saving intermediate checkpoints
            return
        filename = os.path.join(output_dir, filename)
        torch.save(save_dict, filename)


    #=============MAIN TRAINING LOOP=============
    
    #--------------tune cut-------------------
    if args.tuneCut:
        print("Cut tuning started")
        
        #Number of epochs for tuning
        cutEpochs = hparams['tuneCutEpochs']
        print(f"cutEpochs: {cutEpochs}")

        total_steps = cutEpochs * steps_per_epoch #Total tuning steps based on the number of selected epochs

        algorithm.initTuneCut(cutEpochs)
        
        for epoch in range(cutEpochs):
            for step_in_epoch, batch in enumerate(train_loader): #iterate over batches from train_loader
                step = epoch * steps_per_epoch + step_in_epoch #global step
                
                if args.use_masks:
                    i, x, y, a, mask = batch
                    minibatch_device = (i, x.to(device), y.to(device), a.to(device), mask.to(device))
                else:
                    i, x, y, a = batch
                    minibatch_device = (i, x.to(device), y.to(device), a.to(device))
                
                algorithm.update(minibatch_device, step=step, current_epoch=epoch)
    
              
            print(f"[TuneCut] Epoch {epoch+1}/{cutEpochs} completed "
                  f"(Step {step+1}/{total_steps})")
    
        #save tune cuts
        cut, cut2, means, stds = algorithm.returnCut()
        cut_file='cuts.pkl'
        filename = os.path.join(output_dir, cut_file)
        torch.save({"cut": cut, "cut2": cut2, "means": means, "stds": stds}, filename)
        
        algorithm.cut = cut
        algorithm.cut2 = cut2

        print(f"cut: {algorithm.cut}")
        print(f"cut2: {algorithm.cut2}")

        if args.only_tune_cut:
            print("Cutting tuning is complete. Normal training will no longer continue.")
            exit(0)

    #--------------tune cut-------------------

    last_results_keys = None
    step=start_step #global steps counter
    start_epoch=0

    """
    print("DEBUG TRAINING INFO")
    print(f"len(train_dataset) = {len(train_dataset)}")
    print(f"batch_size (hparams) = {hparams['batch_size']}")
    print(f"len(train_loader) = {len(train_loader)}")
    print(f"steps_per_epoch (calculated) = {steps_per_epoch}")
    print(f"num_epochs = {num_epochs}")
    print(f"num_steps = {num_steps}")
    print(f"start_step = {start_step}")
    print(f"start_epoch (before calc) = {start_epoch}")
    if steps_per_epoch > 0:
        print(f"implied epochs from num_steps = {math.ceil(num_steps/steps_per_epoch)}")
    """

    # training time
    training_start_time = time.time()

    #---------training-------------
    for epoch in range(start_epoch, num_epochs): #iterates over epochs (outer loop)
        #print(f"\n>> START EPOCH {epoch}")
        
        if args.use_es and es.early_stop: #If early stopping is used and the stopping condition is met, training stops and the best metric value is displayed.

            #print(f"Early stopping at step {step} with best {args.es_metric}={es.best_score}.")
            print(f"Training stopped. Best {args.es_metric}={es.best_score:.4f} "
            f"at step {es.step} (epoch {es.best_epoch})")
            break #breaks outer loop

        epoch_start_time = time.time() #epochs time
        
        for step_in_epoch, batch in enumerate(train_loader): # Iterate over batches from train_loader (inner loop)
            #print(f"   step_in_epoch={step_in_epoch}", end='')
            step = epoch * steps_per_epoch + step_in_epoch #global step
            #print(f"  -> global step = {step}")
                        
            step_start_time = time.time()
                
            if args.use_masks:
                #i, x, y, a, mask = next(train_minibatches_iterator)  
                i, x, y, a, mask = batch #obtain minibatch
                minibatch_device = (i, x.to(device), y.to(device), a.to(device), mask.to(device)) #move to gpu
            else:    
                #i, x, y, a = next(train_minibatches_iterator)
                i, x, y, a = batch
                minibatch_device = (i, x.to(device), y.to(device), a.to(device)) #move to gpu

            
            algorithm.train() #change to training mode
            step_vals = algorithm.update(minibatch_device, step, epoch) #adjust the model’s weights – update. In step_vals the loss from this step is saved
            checkpoint_vals['step_time'].append(time.time() - step_start_time) #Records the time it took to process this batch in seconds
    
            for key, val in step_vals.items(): # save obteined losses
                checkpoint_vals[key].append(val) 
    
            #Every checkpoint_freq steps, or at the end of training, a checkpoint is saved and the model is evaluated
            if ((not args.debug) and step % checkpoint_freq == 0) or (step == num_steps - 1):
                #print("checkpoint_freq")
                #print(checkpoint_freq)
                results = {
                    'step': step,
                    'epoch': epoch,
                }
                for key, val in checkpoint_vals.items():
                    results[key] = np.mean(val) #the metrics obtained up to the checkpoint are averaged -> stored in checkpoint_vals, which come from step_vals from the update            
                
                #Evaluation on validation and test sets. Returns a dictionary
                curr_metrics = {split: eval_helper.eval_metrics(algorithm, loader, device)
                                for split, loader in zip(split_names, eval_loaders)}
                full_val_metrics = curr_metrics['va'] 
    
                for split in sorted(split_names): #save main metrics
                    results[f'{split}_avg_acc'] = curr_metrics[split]['overall']['accuracy_50']
                    results[f'{split}_overall_auroc'] = curr_metrics[split]['overall']['AUROC']
                    results[f'{split}_worst_acc'] = curr_metrics[split]['min_group']['accuracy_50']
    
                #----print metrics----
                """
                results_keys = list(results.keys())
                if results_keys != last_results_keys:
                    print("\n")
                    misc.print_row([key for key in results_keys if key not in {'mem_gb', 'step_time'}], colwidth=12)
                    last_results_keys = results_keys
                misc.print_row([results[key] for key in results_keys if key not in {'mem_gb', 'step_time'}], colwidth=12)
                """
    
                results_keys = list(results.keys())
                
                print("\n")
                misc.print_row(results_keys, colwidth=12)
                misc.print_row([results[key] for key in results_keys], colwidth=12)

                
                results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.) 
    
                #adds the hparams and args info into results
                results.update({
                    'hparams': hparams,
                    'args': vars(args),
                })
                results.update(curr_metrics)
    
                #save results in a json file
                epochs_path = os.path.join(output_dir, 'results.json')
                with open(epochs_path, 'a') as f:
                    f.write(json.dumps(results, sort_keys=True, cls=misc.NumpyEncoder) + "\n")
    
                # Save the model state and training metadata to a .pkl checkpoint file
                save_dict = {
                    "args": vars(args),
                    "best_es_metric": es.best_score,
                    "epoch": epoch,
                    "start_step": step + 1,
                    "num_labels": num_labels,
                    "num_attributes": train_dataset.num_attributes,
                    "model_input_shape": input_shape,
                    "model_hparams": hparams,
                    "model_dict": algorithm.state_dict(), # stores the model weights
                    #"optimizer_dict": algorithm.optimizer.state_dict(),
                    "early_stopper": es,
                }
                save_checkpoint(save_dict)
    
                if args.use_es:
                    if args.es_strategy == 'metric': #get the validation metric used for early stopping
                        es_metric_val = full_val_metrics[es_group][es_metric]
                    
                    """ 
                    Call the EarlyStopping object
                     This will:
                      - Compare the current metric with the best one so far
                      - Save the model weights if performance improved
                      - Update patience counter if no improvement
                      - Trigger early stopping if patience is exceeded
                    """
                    
                    es(es_metric_val, step, save_dict, best_model_path, steps_per_epoch) 

                    #check if the current metric is the new best score
                    if es.best_score == (-es_metric_val if es.lower_is_better else es_metric_val):
                        print(f"[Epoch {epoch}]  New best {es_metric}: {es_metric_val:.4f}")
                        print(f"   ↳ Best model saved at epoch {es.best_epoch} (step {es.step})")
    
                    if es.early_stop:
                        print(f"⏹️ Early stopping triggered at step {step} (epoch {epoch})")
                        break #breaks inner loop
    
                    
                #Clear metrics
                checkpoint_vals = collections.defaultdict(lambda: [])
                
        # Epoch time
        epoch_time = time.time() - epoch_start_time
        mins, secs = divmod(epoch_time, 60)
        hours, mins = divmod(mins, 60)
        
        print(f"⏱️ Epoch {epoch} -> {int(hours)} h {int(mins)} min {secs:.1f} sec")
        
        #save epcohs time
        results = {
            'epoch': epoch,
            'epoch_time_sec': epoch_time,
            'epoch_time_min': epoch_time / 60,
            'epoch_time_hr': epoch_time / 3600
        }
        
        with open(os.path.join(output_dir, 'epoch_times.json'), 'a') as f:
            f.write(json.dumps(results) + '\n')


    # training time
    
    training_total_time = time.time() - training_start_time
    mins, secs = divmod(training_total_time, 60)
    hours, mins = divmod(mins, 60)
    
    print(f"\n🏁 Training complete in {int(hours)} h {int(mins)} min {secs:.1f} sec")


    #============FINALIZA BUCLE PRINCIPAL DE ENRTENAMIENTO==========================



    #==========Evaluaciond de resultados en validacion y test===============

    # load best model and get metrics on eval sets
    if args.use_es:
        algorithm.load_state_dict(torch.load(os.path.join(output_dir, "model.best.pkl"))['model_dict'])

    algorithm.eval() #algorithm in eval mode

    #eval dataloaders
    split_names = ['va', 'te']
    final_eval_loaders = [DataLoader(
        dataset=dset,
        batch_size=max(128, hparams['batch_size'] * 2),
        num_workers=num_workers,
        shuffle=False)
        for dset in [val_dataset, test_dataset]
    ]


    final_results = {split: eval_helper.eval_metrics(algorithm, loader, device, add_arrays=True)
                     for split, loader in zip(split_names, final_eval_loaders)}
    if args.use_es:
        final_results['es_step'] = es.step

    # save all final evaluation metrics 
    pickle.dump(final_results, open(os.path.join(output_dir, 'final_results.pkl'), 'wb'))

    # print test accuracy summary (threshold 0.5) using the best validation checkpoint
    print("\nTest accuracy (best validation checkpoint):")
    print(f"\tmean:\t[{final_results['te']['overall']['accuracy_50']:.3f}]\n"
          f"\tworst:\t[{final_results['te']['min_group']['accuracy_50']:.3f}]")
    
    # group-wise accuracy metrics
    # Prints the model accuracy for each demographic group or subgroup in va and te sets
    # accuracy_50: Accuracy computed using a 50% threshold to classify predictions as positive or negative

    print("Group-wise accuracy:")
    for split in final_results.keys():
        if split != 'es_step' and not split.startswith('lin_'):
            print('\t[{}] group-wise {}'.format(
                split, (np.array2string(
                    pd.DataFrame(final_results[split]['per_group']).T['accuracy_50'].values,
                    separator=', ', formatter={'float_kind': lambda x: "%.3f" % x}))))

    
    # Create a marker file to indicate that training and evaluation have successfully completed
    with open(os.path.join(output_dir, 'done'), 'w') as f:
        f.write('done')




