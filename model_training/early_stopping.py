#early_stopping.py

import torch

# Indicates whether a lower metric value is better.
# Ex: AUROC/accuracy: higher is better (False).
# Ex: BCE/FPR: lower is better (True).

lower_is_better = {
    'AUROC': False,
    'AUPRC': False,
    'BCE': True,
    'ECE': True,
    'accuracy': False,
    'balanced_accuracy': False,
    'precision': False,
    'TPR': False,
    'TNR': False,
    'FPR': True,
    'FNR': True
}




class EarlyStopping:

    def __init__(self, patience=5, lower_is_better=True, min_delta=0.0):
        self.patience = patience # Max steps allowed without meaningful improvement
        self.counter = 0 # Consecutive steps without improvement
        self.best_score = None  # Best metric value seen so far
        self.early_stop = False # Flag to stop training
        self.step = 0 # save the step at which best metric occurred
        self.best_epoch = 0  # save the epoch of the best model
        self.min_delta = min_delta # Minimum improvement required to count as progress
        self.lower_is_better = lower_is_better # True if lower metric is better (e.g., loss)
       

    # metric: current metric value
    # step: current training step
    # state_dict: model parameters
    # path: save location
    # steps_per_epoch: steps in one epoch (optional)

    def __call__(self, metric, step, state_dict, path, steps_per_epoch=None):
        if self.lower_is_better: # Invert metric if lower is better
            score = -metric
        else:
            score = metric

      
        if self.best_score is None:  # First metric — initialize best score
            self.best_score = score
            self.step = step
            self.last_score = score
            self.last_state_dict = state_dict
            
            if steps_per_epoch is not None:
                self.best_epoch = step // steps_per_epoch
            save_model(state_dict, path) #save model

        #save the last value for final compariso
        self.last_score = score
        self.last_state_dict = state_dict
        
        #elif score <= self.best_score:
        if (score - self.best_score) <= self.min_delta: #Check if the improvement is smaller than min_delta
            self.counter += 1
           
            if self.counter >= self.patience:
                if self.last_score > self.best_score:
                    print("⚠️ The last model is slightly better — saving it as the final best model.")
                    save_model(self.last_state_dict, path)
                    self.best_score = self.last_score
                self.early_stop = True #trigger early stopping when patience is exceeded
                
        else: # Performance improved: save model and reset counter
            save_model(state_dict, path)
            self.best_score = score 
            self.step = step
            if steps_per_epoch is not None:
                self.best_epoch = step // steps_per_epoch 
            self.counter = 0 #reset counter



def save_model(state_dict, path): # Save the model parameters to the specified path
    torch.save(state_dict, path)



