#experiments.py


import os
import json
from pathlib import Path
from itertools import product


# Returns a list of all possible combinations from a flat grid (dictionary)
def combinations_base(grid):
    return list(dict(zip(grid.keys(), values)) for values in product(*grid.values()))


# Generates all parameter combinations, supporting nested sub-experiments
def combinations(grid):
    sub_exp_names = set()
    args = []
    for i in grid:
        if isinstance(grid[i], dict):
            for j in grid[i]:
                sub_exp_names.add(j)
    if len(sub_exp_names) == 0:
        return combinations_base(grid)

    for i in grid: # Ensure all nested dicts share the same sub-experiment names
        if isinstance(grid[i], dict):
            assert set(list(grid[i].keys())) == sub_exp_names, f'{i} does not have all sub exps ({sub_exp_names})'
    
    for n in sub_exp_names: # Build combinations for each sub-experiment
        sub_grid = grid.copy()
        for i in sub_grid:
            if isinstance(sub_grid[i], dict):
                sub_grid[i] = sub_grid[i][n]
        args += combinations_base(sub_grid)
    return args
    

# Extracts the hyperparameters for the given experiment class
def get_hparams(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment]().get_hparams()


# Returns the script file name (e.g., "train", "eval") associated with the experiment class
def get_script_name(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].fname




#===========================================================================================
#==================================GRIDS===ERM========================================
#===========================================================================================


# ========== training ========== #

#main experiment grids, 12 hparams seeds
class grid_race_mimic_12:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['MIMIC'],
            'algorithm': ['ERM'],
            'task': ['No Finding', 'Cardiomegaly', 'Pleural Effusion', 'Pneumothorax'],
            'attr': ['ethnicity'],
            'hparams_seed': list(range(12)),
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0, 1, 2],
            "image_arch": ["densenet121"],
            "data_dir": ["/home/lchanch/initial_training/image_df"],
            "min_delta": [0.001],
            "checkpoint_freq": [1000],
            "skip_ood_eval": [True],
            "skip_model_save": [True],
        }

    def get_hparams(self):
        return combinations(self.hparams)

 
#5 final independet runs with best hyperparameter seed
class grid_race_mimic_final:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['MIMIC'],
            'algorithm': ['ERM'],
            'task': ['No Finding', 'Cardiomegaly', 'Pleural Effusion', 'Pneumothorax'],
            'attr': ['ethnicity'],
            'hparams_seed': [9], #best hyperparameter seed
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0, 1, 2, 3, 4, 5],
            "image_arch": ["densenet121"],
            "data_dir": ["/home/lchanch/initial_training/image_df"],
            "min_delta": [0.001],
            "checkpoint_freq": [1000],
            "skip_ood_eval": [True],
            "skip_model_save": [True],
        }

    def get_hparams(self):
        return combinations(self.hparams)


# ========== training ========== #

# ========== evaluation ========== #

# returns a list of all models to evaluate, based on the experiments marked as "done"
class eval_race_mimic_all:
    fname = 'eval'
    root_dir = Path('/home/lchanch/initial_training/output_sweep_12/grid_sex_mimic_final')

    def __init__(self):
        dirs = []
        for i in self.root_dir.glob('**/done'):
            if not (i.parent/'done_eval').is_file():
                dirs.append(str(i.parent))

        self.hparams = {
            'model_dir': dirs,
        }

    def get_hparams(self):
        print(f"Dirs: {len(self.hparams['model_dir'])}") 
        print(f"Combinaciones: {len(combinations(self.hparams))}")
        return combinations(self.hparams)

# ========== evaluation ========== #


#===========================================================================================
#==================================GRIDS===ERM========================================
#===========================================================================================



#===========================================================================================
#==================================GRIDS===ISNetDANN========================================
#===========================================================================================

# ========== training ========== #

#main experiment grids, 12 hparams seeds
class grid_sex_ISNetDANN:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['MIMIC'],
            'algorithm': ['ISNetDANN'],
            'task': ['No Finding'],
            'attr': ['sex'],
            'hparams_seed': [0,11], # y <- 1,2,3,4,5,6,7,8,9,10,11,0
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0, 1, 2],
            "epochs": [10],
            "image_arch": ["densenet121"],
            #"data_dir": ["/home/lchanch/initial_training/LRP/mask_recon/image_mask_reduced_df"],
            "data_dir": ["/home/lchanch/df_construction_mapping/image_mask_df"],
            "min_delta": [0.001],
            "checkpoint_freq": [1000],
            "skip_model_save": [True],
            'use_masks': [True],
        }

    def get_hparams(self):
        return combinations(self.hparams)

#main experiment grids, 12 hparams seeds
class grid_race_ISNetDANN_:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['MIMIC'],
            'algorithm': ['ISNetDANN'],
            'task': ['No Finding'],
            'attr': ['race'],
            'hparams_seed': [1], # y <- 1,2,3,4,5,6,7,8,9,10,11,0
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0, 1, 2],
            "epochs": [10],
            "image_arch": ["densenet121"],
            "data_dir": ["/home/lchanch/df_construction_mapping/image_mask_df"],
            "min_delta": [0.001],
            "checkpoint_freq": [1000],
            "skip_model_save": [True],
            'use_masks': [True],
        }

    def get_hparams(self):
        return combinations(self.hparams)


#5 final independet runs with best hyperparameter seed
class grid_sex_ISNetDANN_final:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['MIMIC'],
            'algorithm': ['ISNetDANN'],
            'task': ['No Finding'],
            'attr': ['sex'],
            'hparams_seed': [8], #best hyperparameter seed
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0,1,0,3,4,5],
            "epochs": [10],
            "image_arch": ["densenet121"],
            "data_dir": ["/home/lchanch/df_construction_mapping/image_mask_df"],
            "min_delta": [0.001],
            "checkpoint_freq": [1000],
            "skip_model_save": [True],
            'use_masks': [True],
        }

    def get_hparams(self):
        return combinations(self.hparams)

# ========== training ========== #

# ========== evaluation ========== #

# returns a list of all models to evaluate, based on the experiments marked as "done"
class eval_sex_ISNetDANN:
    fname = 'eval'
    root_dir = Path('/home/lchanch/model_training/ISNetDANN/train/grid_sex_ISNetDANN_hp8_final')

    def __init__(self):
        dirs = []
        for i in self.root_dir.glob('**/done'):
            if not (i.parent/'done_eval').is_file():
                dirs.append(str(i.parent))

        self.hparams = {
            'model_dir': dirs,
        }

    def get_hparams(self):
        print(f"Dirs: {len(self.hparams['model_dir'])}") 
        print(f"Combinaciones: {len(combinations(self.hparams))}")
        return combinations(self.hparams)

# ========== evaluation ========== #

#===========================================================================================
#==================================GRIDS===ISNetDANN========================================
#===========================================================================================



#====================Dummy grid for ISNet======================

class grid_dummy_ISNetDANN:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['MIMIC'],
            'algorithm': ['ISNetDANN'],
            'task': ['No Finding'],
            'attr': ['sex'],
            'hparams_seed': [0], #best hyperparameter seed
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0],
            "epochs": [3],
            "image_arch": ["densenet121"],
            "data_dir": ["/home/lchanch/initial_training/LRP/mask_recon/image_mask_reduced_df"],
            "min_delta": [0.001],
            "checkpoint_freq": [1000],
            "skip_model_save": [True],
            'use_masks': [True],
        }

    def get_hparams(self):
        return combinations(self.hparams)


# returns a list of all models to evaluate, based on the experiments marked as "done"
class eval_dummy_ISNetDANN:
    fname = 'eval'
    root_dir = Path('/home/lchanch/model_training/ISNetDANN/dummy_exp/grid_dummy_ISNetDANN')

    def __init__(self):
        dirs = []
        for i in self.root_dir.glob('**/done'):
            if not (i.parent/'done_eval').is_file():
                dirs.append(str(i.parent))

        self.hparams = {
            'model_dir': dirs,
        }

    def get_hparams(self):
        print(f"Dirs: {len(self.hparams['model_dir'])}") 
        print(f"Combinaciones: {len(combinations(self.hparams))}")
        return combinations(self.hparams)

#====================Dummy grid for ISNet======================