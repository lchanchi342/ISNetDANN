#eval.py

#Loads a trained model
#Evaluates performance across datasets, attributes, and groups
#Computes fairness metrics and performance gaps
#Extracts representations and performs linear evaluation


import argparse
import json
import os
import random
import sys
import time
import numpy as np
import PIL
import torchvision
import torch.utils.data
import pickle
from pathlib import Path

from torch.utils.data import DataLoader

from model_training import datasets, algorithms
from model_eval import eval_helper, lin_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluating a trained model on all datasets')
    parser.add_argument('--model_dir', type=str, required=True) #Path to the directory containing the trained model
    parser.add_argument('--opt_thres_file', type=str,
                        default=Path('/home/lchanch/initial_training/output_sweep_12/opt_thres_sex_mimic_12.pkl')) #Path to the directory containing the opt_thres file .pkl
    #/home/lchanch/model_training/ISNetDANN/eval/eval_sex/opt_thres_sex_ISNetDANN.pkl
    parser.add_argument('--data_dir', type=str, default=Path('/home/lchanch/df_construction_mapping/image_df')) #Directory containing dataset metadata
    #data dir only masks: /home/lchanch/df_construction_mapping/image_mask_df
    #metadata without masks: /home/lchanch/df_construction_mapping/image_df
    
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Print environment information
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    # Print input arguments
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.multiprocessing.set_sharing_strategy('file_system')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load trained model and hyperparameters
    model_path = Path(args.model_dir)
    output_dir = model_path
    old_args = json.load((model_path/'args.json').open('r'))#load args from the model -> old args
    loaded = torch.load(model_path/'model.best.pkl')#load best model
    hparams = loaded['model_hparams'] #load hyperparameters
    dataset = old_args['dataset']


    #print loaded args (old args) -> from the model
    print('Training Args:')
    for k, v in sorted(old_args.items()):
        print('\t{}: {}'.format(k, v))

    #Load optimal thresholds
    opt_thress = pickle.load(Path(args.opt_thres_file).open('rb'))
    opt_thres = opt_thress[(dataset[0], hparams['task'], hparams['attr'], old_args['algorithm'])]

    #Evaluate using both standard (0.5) and optimal thresholds
    eval_thress = [0.5, opt_thres]
    eval_thress_suffix = ['_50', '_opt']


    #Dataset loading
    if len(dataset) == 1:
        if dataset[0] in vars(datasets):
          
            train_dataset = vars(datasets)[dataset[0]](args.data_dir, 'tr', hparams, group_def='group')
            val_dataset = vars(datasets)[dataset[0]](args.data_dir, 'va', hparams, group_def='group')
            test_dataset = vars(datasets)[dataset[0]](args.data_dir, 'te', hparams, group_def='group')
        
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
   
    #Rebuild model architecture, with same hyperameters and load trained weights (model_dict)
    algorithm_class = algorithms.get_algorithm_class(old_args['algorithm'])
    algorithm = algorithm_class('images', (3, 256, 256), 2, loaded['num_attributes'], 0, hparams,
                                grp_sizes=train_dataset.group_sizes, attr_sizes=train_dataset.attr_sizes).to(device)
    algorithm.load_state_dict(loaded['model_dict'])

    algorithm.eval() #eval mode

    
    #Dataloaders
    num_workers = train_dataset.N_WORKERS

    split_names = ['va', 'te']
    final_eval_loaders = [DataLoader(
        dataset=dset,
        batch_size=max(128, hparams['batch_size'] * 2),
        #batch_size=max(64, hparams['batch_size'] * 2),
        num_workers=num_workers,
        shuffle=False)
        for dset in [val_dataset, test_dataset]
    ]

    # Evaluate on all compatible CXR datasets and attributes
    if dataset[0] in datasets.CXR_DATASETS:
        # all CXR datasets with matching tasks
        all_cxr = []
        for ds in datasets.CXR_DATASETS:
            if hparams['task'] in vars(datasets)[ds].TASKS:
                all_cxr.append(ds)
                split_names_ds = vars(datasets)[ds].EVAL_SPLITS
                for attr in vars(datasets)[ds].AVAILABLE_ATTRS:
                    final_eval_loaders += [DataLoader(
                        dataset=dset,
                        batch_size=max(128, hparams['batch_size'] * 2),
                        #batch_size=max(64, hparams['batch_size'] * 2),
                        num_workers=num_workers,
                        shuffle=False)
                        for dset in [vars(datasets)[ds](args.data_dir, split, hparams, override_attr=attr)
                                     for split in split_names_ds]
                    ]
                    for j in split_names_ds:
                        split_names.append(f'{ds}-{attr}-{j}')
        
    print("Before eval on all sets", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    #Model evaluation (performance + fairness metrics) - uses eval helper - uses th 2 thresholds:  opt y 0.5
    final_results = {split: eval_helper.eval_metrics(algorithm, loader, device, add_arrays=True, thress=eval_thress,
                                                     thress_suffix=eval_thress_suffix)
                     for split, loader in zip(split_names, final_eval_loaders)}

    print("Finished eval; Starting representation computation", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))


    # get representations (features) and train linear predictors
    
    train_dataset_rep = train_dataset 
    
    train_zs, train_atts, train_ys = lin_eval.get_representations(algorithm, DataLoader(
        dataset=train_dataset_rep,
        batch_size=max(64, hparams['batch_size'] * 2),
        #batch_size=max(128, hparams['batch_size'] * 2),
        num_workers=num_workers,
        shuffle=False), device)
    
    val_zs, val_atts, val_ys = lin_eval.get_representations(algorithm, final_eval_loaders[0], device)
    test_zs, test_atts, test_ys = lin_eval.get_representations(algorithm, final_eval_loaders[1], device)

    #save extracter representaions (features) in reps.pkl
    pickle.dump({
         'va': (val_zs, val_atts, val_ys),
         'te': (test_zs, test_atts, test_ys)
    }, open(os.path.join(output_dir, 'reps.pkl'), 'wb'))

    print("Finished representation computation; Starting LR training", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    # Linear evaluation for attribute prediction, uses the features - uses lin_eval
    lin_eval_metrics = lin_eval.eval_lin_attr_pred(train_zs, train_atts, train_ys,
                                                   val_zs, val_atts, val_ys,
                                                   test_zs, test_atts, test_ys)

    # Save evaluation metrics, linear regressor results, and the selected threshold to final_results_eval.pkl
    final_results = {**final_results, **lin_eval_metrics, 'opt_thres': opt_thres}
    pickle.dump(final_results, open(os.path.join(output_dir, 'final_results_eval.pkl'), 'wb'))

    # Mark evaluation as completed
    with open(os.path.join(output_dir, 'done_eval'), 'w') as f:
        f.write('done')

    print("Finished everything :) !", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))



