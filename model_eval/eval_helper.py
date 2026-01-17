#eval_helper.py

# predict_on_set -> obtains model predictions for each sample in the batch
# prob_metrics -> Compute probability and calibration metrics to evaluate the performance of a classification model
# binary_metrics -> Computes binary classification metrics (accuracy, sensitivity, specificity, etc.) FPR, FNR


import torch 
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, average_precision_score,
                             balanced_accuracy_score, recall_score, brier_score_loss, log_loss, classification_report)
import netcal.metrics


def predict_on_set(algorithm, loader, device):
    num_labels = loader.dataset.num_labels
    ys, atts, gs, ps = [], [], [], []

    algorithm.eval()
    #with torch.no_grad(): # grads required for LRP
   
    for batch in loader:
        if len(batch) == 5:
            _, x, y, a, _ = batch  # does not consider the mask
        else:
            _, x, y, a = batch
            
        p = algorithm.predict(x.to(device))
        
        if isinstance(p, dict): #LRP returns a dictionary with the results
            p = p['output'] #['output'] to extract only the logits; in LRP the heatmaps are also returned
         
        if p.squeeze().ndim == 1: #Applies sigmoid or softmax depending on the number of classes.
            #print(">>>>Applying Sigmoid")
            p = torch.sigmoid(p).detach().cpu().numpy()
        else:
            # print(">>>>Applying Softmax")
            # one logit per class (out_features: 2), so softmax is used
            # softmax is used because cross-entropy loss is applied
            p = torch.softmax(p, dim=-1).detach().cpu().numpy()
            if num_labels == 2:
                p = p[:, 1] # takes the probability of label 1 (i.e., presence of the pathology)

        ps.append(p) #predicts
        ys.append(y) #labels
        atts.append(a) #attr demo
        gs.append([f'y={yi},a={gi}' for c, (yi, gi) in enumerate(zip(y, a))]) # grupos (label + atributo)
        # gs -> Label–attribute pairs encoded as strings (e.g., y=1,a=0) for subgroup analysis

    return np.concatenate(ys, axis=0), np.concatenate(atts, axis=0), np.concatenate(ps, axis=0), np.concatenate(gs)



# Computes standard (overall) performance metrics, overall, per class, per attribute, and per group

def eval_metrics(algorithm, loader, device, thress=[0.5], thress_suffix=['_50'], add_arrays=False):
    targets, attributes, preds, gs = predict_on_set(algorithm, loader, device) #Obtain predictions
    label_set = np.unique(targets)

    preds_rounded = {suffix: preds >= thres if preds.squeeze().ndim == 1 else preds.argmax(1)   # apply threshold: e.g >= 0.5 -> class 1, < 0.5 -> class 0
                     for thres, suffix in zip(thress, thress_suffix)}

    # GLOBAL metrics computation
    res = {}
    res['overall'] = prob_metrics(targets, preds, label_set)# Computes probabilistic metrics (prob_metrics): BCE, ECE, AUROC, AUPRC

    for thres, suffix in zip(thress, thress_suffix):
        res['overall'] = {**res['overall'], **binary_metrics(targets, preds_rounded[suffix], label_set, suffix=suffix)}  # Merge probabilistic metrics with binary classification metrics

    # Metrics PER DEMOGRAPHIC ATTRIBUTE
    res['per_attribute'] = {}
    res['per_class'] = {}
    res['per_group'] = {}

    # Per-attribute results
    # Computes probabilistic and binary metrics for each attribute subgroup
    for a in np.unique(attributes):
        mask = attributes == a
        res['per_attribute'][int(a)] = prob_metrics(targets[mask], preds[mask], label_set) # Filter samples by attribute

        for thres, suffix in zip(thress, thress_suffix):
            res['per_attribute'][int(a)] = {**res['per_attribute'][int(a)], **binary_metrics(
                targets[mask], preds_rounded[suffix][mask], label_set, suffix=suffix)}

    # Per-class binary results
    # Uses classification_report to obtain precision, recall, and F1-score per class
    for thres, suffix in zip(thress, thress_suffix):
        classes_report = classification_report(targets, preds_rounded[suffix], output_dict=True, zero_division=0.)
        res['overall'][f'macro_avg_{suffix}'] = classes_report['macro avg']
        res['overall'][f'weighted_avg_{suffix}'] = classes_report['weighted avg']
        for y in np.unique(targets):
            res['per_class'][int(y)] = {f'{i}{suffix}': classes_report[str(y)][i] for i in classes_report[str(y)]}

    # Per-class AUROC
    # Computes AUROC for each class
    if preds.squeeze().ndim == 1:  # 2 classes
        res['per_class'][1]['AUROC'] = roc_auc_score(targets, preds, labels=[0, 1])
        res['per_class'][0]['AUROC'] = res['per_class'][1]['AUROC']
    else:
        for y in np.unique(targets): #multiclass
            new_label = targets == y
            new_preds = preds[:, int(y)]
            res['per_class'][int(y)]['AUROC'] = roc_auc_score(new_label, new_preds, labels=[0, 1])

    # Per-group binary results
    # Filters data by subgroup (g) and computes binary metrics
    for g in np.unique(gs):
        mask = gs == g
        res['per_group'][g] = {}
        for thres, suffix in zip(thress, thress_suffix):
            res['per_group'][g] = {
                **res['per_group'][g],
                **binary_metrics(targets[mask], preds_rounded[suffix][mask], label_set, suffix=suffix)
            }

    #for GAPs analysis 

    # res['adjusted_accuracy'] = sum([res['per_group'][g]['accuracy'] for g in np.unique(gs)]) / len(np.unique(gs))
    res['min_attr'] = pd.DataFrame(res['per_attribute']).min(axis=1).to_dict() #min_attr / max_attr: worst and best performance across demographic attributes
    res['max_attr'] = pd.DataFrame(res['per_attribute']).max(axis=1).to_dict()
    res['min_group'] = pd.DataFrame(res['per_group']).min(axis=1).to_dict() #min_group / max_group: worst and best performance across (label, attribute) groups
    res['max_group'] = pd.DataFrame(res['per_group']).max(axis=1).to_dict()
    res['max_gap'] = (pd.DataFrame(res['per_attribute']).max(axis=1) - pd.DataFrame(res['per_attribute']).min(axis=1)).to_dict() # Gap between the best- and worst-performing attribute subgroups
    

    if add_arrays:
        res['y'] = targets
        res['a'] = attributes
        res['preds'] = preds

    return res




# Computes binary classification metrics (accuracy, sensitivity, specificity, etc.)
# Computes FPR, FNR and prevalences for fairness gap analysis 

def binary_metrics(targets, preds, label_set=[0, 1], suffix='', return_arrays=False):
    if len(targets) == 0:
        return {}

    res = {
        'accuracy': accuracy_score(targets, preds),
        'n_samples': len(targets)
    }

    if len(label_set) == 2:
        CM = confusion_matrix(targets, preds, labels=label_set)

        res['TN'] = CM[0][0].item() #True Negatives
        res['FN'] = CM[1][0].item() #False Negatives
        res['TP'] = CM[1][1].item() #True Positives
        res['FP'] = CM[0][1].item() #False Positives

        res['error'] = res['FN'] + res['FP']   #Total number of errors

        # True Positive Rate (TPR) and False Negative Rate (FNR)
        if res['TP'] + res['FN'] == 0:
            res['TPR'] = 0
            res['FNR'] = 1
        else:
            res['TPR'] = res['TP']/(res['TP']+res['FN']) # Sensitivity / Recall
            res['FNR'] = res['FN']/(res['TP']+res['FN']) # False Negative Rate

        if res['FP'] + res['TN'] == 0:
            res['FPR'] = 1 
            res['TNR'] = 0
        else:
            res['FPR'] = res['FP']/(res['FP']+res['TN']) # False Positive Rate
            res['TNR'] = res['TN']/(res['FP']+res['TN']) # Specificity

        res['pred_prevalence'] = (res['TP'] + res['FP']) / res['n_samples'] # Fraction of samples predicted as positive.
        res['prevalence'] = (res['TP'] + res['FN']) / res['n_samples'] #Fraction of samples that are actually positive.
    else: #for multiclass-> only TPR recall
        CM = confusion_matrix(targets, preds, labels=label_set)
        res['TPR'] = recall_score(targets, preds, labels=label_set, average='macro', zero_division=0.)

    # Balanced accuracy computation
    if len(np.unique(targets)) > 1:
        res['balanced_acc'] = balanced_accuracy_score(targets, preds)

    if return_arrays:
        res['targets'] = targets
        res['preds'] = preds

    return {f"{i}{suffix}": res[i] for i in res} # Add suffix to metric names




# Compute probability and calibration metrics to evaluate the performance of a classification model
# BCE: Binary Cross Entropy (log-loss), measures the difference between predicted probabilities and true labels.
# ECE: Expected Calibration Error, measures how well the predicted probabilities reflect reality.

def prob_metrics(targets, preds, label_set, return_arrays=False):
    if len(targets) == 0:
        return {}

    res = {
        'BCE': log_loss(targets, preds, labels=label_set),
        'ECE': netcal.metrics.ECE().measure(preds, targets)
    }

    if len(set(targets)) > 2:
        # happens when you predict a class, but there are no samples with that class in the dataset
        try:
            res['AUROC'] = roc_auc_score(targets, preds, multi_class='ovr', labels=label_set) # ovr: One-vs-Rest (one class against all others)
        except:
            res['AUROC'] = roc_auc_score(targets, preds, multi_class='ovo', labels=label_set) # ovo: One-vs-One (each class compared against every other class)


    # If there are only two classes, compute AUROC standard
    # AUROC: Area Under the Receiver Operating Characteristic curve.
    elif len(set(targets)) == 2:
        res['AUROC'] = roc_auc_score(targets, preds, labels=label_set)

    elif len(set(targets)) == 1:
        res['AUROC'] = None

    # Additional metrics for binary classification
    if len(set(targets)) == 2:
        # res['ROC_curve'] = roc_curve(targets, preds)
        res['AUPRC'] = average_precision_score(targets, preds, average='macro') # Area under the Precision-Recall curve
        res['brier'] = brier_score_loss(targets, preds) # Brier Score, measures the quality of predicted probabilities
        res['mean_pred_1'] = preds[targets == 1].mean() # Mean predicted probability for the positive class (1)
        res['mean_pred_0'] = preds[targets == 0].mean() # Mean predicted probability for the negative class (0)

    if return_arrays:
        res['targets'] = targets
        res['preds'] = preds

    return res

