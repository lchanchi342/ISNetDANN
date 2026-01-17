#algorithm.py

#Available algorithms: ERM, ISNetAlgorithms, DANN, CDANN, ISNetDANN

#base class: Algorithm, subclasses must implement:
#loss computation
#parameter updates
#feature extraction 
#prediction


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import copy
import numpy as np
from transformers import get_scheduler

from model_training import networks
from model_training.optimizers import get_optimizers

from model_training.LRP import ISNetFlexTorch


ALGORITHMS = [
    'ERM',
    'ISNetAlgorithm',
    'DANN',
    'CDANN',
    'ISNetDANN'
    ]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


#=========================ALGORITHM -> Base class===========================

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a subgroup robustness algorithm.
    Subclasses should implement the following:
    - _init_model()
    - _compute_loss()
    - update()
    - return_feats()
    - predict()
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None, attr_sizes=None):
        super(Algorithm, self).__init__() # Calls the constructor of the base class of PyTorch torch.nn.Module.
        self.hparams = hparams
        self.data_type = data_type
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.num_examples = num_examples

    def _init_model(self):
        raise NotImplementedError

    def _compute_loss(self, i, x, y, a, step):
        raise NotImplementedError

    def update(self, minibatch, step, current_epoch):
        """Perform one update step."""
        raise NotImplementedError

    def return_feats(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def return_groups(self, y, a): # Groups samples by (class, attribute). Returns all possible combinations of class (has the pathology or not) with the sensitive attribute (e.g., sex: F, M).

        """Given a list of (y, a) tuples, return indexes of samples belonging to each subgroup"""
        idx_g, idx_samples = [], []
        all_g = y * self.num_attributes + a

        for g in all_g.unique():
            idx_g.append(g)
            idx_samples.append(all_g == g)

        return zip(idx_g, idx_samples)

    @staticmethod
    def return_attributes(all_a):# Groups samples by attributes only, egardless of their class.
        """Given a list of attributes, return indexes of samples belonging to each attribute"""
        idx_a, idx_samples = [], []

        for a in all_a.unique():
            idx_a.append(a)
            idx_samples.append(all_a == a)

        return zip(idx_a, idx_samples)


#=========================ALGORITHM -> Base class===========================



#==================================ERM======================================

#ERM - Empirical Risk Minimization (Standard training)

class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None, attr_sizes=None):
        super(ERM, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes, attr_sizes)

        self.featurizer = networks.Featurizer(data_type, input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, # uses the number of outputs from the Featurizer
            num_classes,
            self.hparams['nonlinear_classifier']
        )
        self.network = nn.Sequential(self.featurizer, self.classifier) # combines the feature extractor and the classifier into a sequential network

        self._init_model()

    def _init_model(self):
        self.clip_grad = False

        if self.data_type in ["images", "tabular"]:
            self.optimizer = get_optimizers[self.hparams['optimizer']](
                self.network,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = None
            self.loss = torch.nn.CrossEntropyLoss(reduction="none") # defines the loss function as cross-entropy without reduction
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")

    def _compute_loss(self, i, x, y, a, step):
        return self.loss(self.predict(x), y).mean()

    def update(self, minibatch, step, current_epoch): #important: performs the model's training step

        all_i, all_x, all_y, all_a = minibatch
        loss = self._compute_loss(all_i, all_x, all_y, all_a, step) #compute minibatch loss

        self.optimizer.zero_grad() #reset accumulated gradients
        
        loss.backward() #backpropagate to compute gradients

        self.optimizer.step() #update model parameters

        if self.lr_scheduler is not None:
            self.lr_scheduler.step() #update learning rate


        return {'loss': loss.item()} #converts the loss tensor into a numeric value and returns it in a dictionary

    def return_feats(self, x):
        return self.featurizer(x) #returns the extracted features

    def predict(self, x):
        return self.network(x) 
        # network is the model built in the constructor (featurizer + classifier), so this returns the final predictions



#=========================ERM===========================


#=========================ISNet===========================

class ISNetAlgorithm(Algorithm): 
    """ISNet - Implicit Segmentaion Network"""
    
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, nesterov=False, dropLr=None, val='ISNet', explainLabels=False, grp_sizes=None, attr_sizes=None ):
         
        super(ISNetAlgorithm, self).__init__(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes, attr_sizes)
        
        self.network = networks.Complete_Net(input_shape, self.hparams) # creation of the full network, including Featurizer and Classifier

        
        self.heat=self.hparams['heat'] # default: True, activate heatmap analysis
        self.lr=self.hparams['lr']
        self.optimizer=self.hparams['optimizer']
        self.momentum=self.hparams['momentum']
        self.WD=self.hparams['weight_decay']
        self.nesterov=nesterov 
        self.clip=self.hparams['clip']
        self.dropLr=dropLr 
        self.penalizeAll=self.hparams['penalizeAll']
        self.dLoss=self.hparams['dLoss']
        self.testLoss=False 
        self.alternativeForeground=self.hparams['alternativeForeground']
        self.norm=self.hparams['norm']
        self.val=val 
        #self.explainLabels=explainLabels

        self.Phparams=self.hparams['P'] #Explainability weighting parameter

        # checks whether P from hparams is a dictionary or a fixed value
        if isinstance(self.Phparams, dict): 
            self.P=self.Phparams[0] 
            self.increaseP=self.Phparams 
            
        else:
            self.P=self.Phparams 
            self.increaseP=None
        
        self.d=self.hparams['d']
        
        self.cut=self.hparams['cut']
        self.cut2=self.hparams['cut2']
        self.cutEpochs = self.hparams['tuneCutEpochs']
        self.tuneCut = self.hparams.get('tuneCut', False)
        self.A=self.hparams['A']
        self.B=self.hparams['B']
        self.E=self.hparams['Ea']

        self._init_model()

    def _init_model(self): 

        self.optimizer = get_optimizers[self.hparams['optimizer']]( #sgd
            self.network,
            lr=self.hparams["lr"], 
            weight_decay=self.hparams['weight_decay'], 
            momentum=self.hparams['momentum'], #0.9
            nesterov=self.nesterov) #default: False
        
        self.lr_scheduler = None
       
        if self.dropLr is not None:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.dropLr,
                verbose=True
            )
        else:
            self.scheduler = None

    def _compute_loss(self, i, x, y, a, step, masks=None, norm=None): # Compute ISNet-specific loss

        out = self.predict(x) #Forward pass
        
        if norm is None: 
            norm=self.norm
        
        #calls CompoundLoss from ISNetFlexTorch, which returns the computed losses
        loss = ISNetFlexTorch.CompoundLoss(out=out,
                                         labels=y,
                                         masks=masks,
                                         tuneCut=self.tuneCut, #Default: False
                                         d=self.d,
                                         dLoss=self.dLoss,
                                         cutFlex=self.cut,
                                         cut2Flex=self.cut2,
                                         A=self.A,
                                         B=self.B,
                                         E=self.E,
                                         alternativeForeground=self.alternativeForeground,
                                         norm=norm)
    
        if not self.heat or masks is None: #if heat is False or no masks are provided, return only the classification loss
            return loss['classification']
        if not self.tuneCut: #if heat is True, masks True, and tuneCut is False → standard ISNet with the two losses: heatmap loss and classification loss
            return loss['classification'],loss['LRPFlex']
        else:
            self.keys=list(loss['mapAbsFlex'].keys()) # if heat is True, masks exist, and tuneCut is True
            return loss['classification'],loss['LRPFlex'],loss['mapAbsFlex']

    def update(self, minibatch, step, current_epoch):  #performs the model's training step
        """Perform one update step."""
        
        self.optimizer.zero_grad() #Clear gradients
        
        all_i, all_x, all_y, all_a, all_mask = minibatch
        
        self.step=step
        self.current_epoch= current_epoch

        if self.tuneCut: 
            #print("[DEBUG] tuneCut")

            if self.current_epoch < self.cutEpochs - 1:
                self.heat=False                
                loss = self._compute_loss(all_i, all_x, all_y, all_a, step, masks=all_mask)

            elif self.current_epoch == self.cutEpochs - 1:
                #print(f"[DEBUG] Entering last tuning epoch: {self.current_epoch}/{self.cutEpochs-1}")
                # Last tuning epoch
                self.heat=True

                cLoss, hLoss, mapAbs = self._compute_loss(all_i, all_x, all_y, all_a, step, masks=all_mask)
                #take only values from last tuning epoch
                self.updateCut(mapAbs)
                #use only for tuning cut value, ignores heatmap loss
                loss=cLoss
        
        else: #NORMAL TRAINING
            
            #If P from hparams is not none checks whether is a dictionary or a fixed value
            if (self.increaseP is not None):
                epochs=list(self.increaseP.keys()) # increaseP stores the epochs where P should be increased when a dict is provided
                epochs.sort()
                for epoch in epochs:
                    if (self.current_epoch>=epoch):
                        self.P=self.increaseP[epoch]

            if (self.heat):#ISNet #NORMAL TRAINING
                
                cLoss,hLoss=self._compute_loss(all_i, all_x, all_y, all_a, step,
                                               masks=all_mask)
                
                loss=(1-self.P)*cLoss+self.P*hLoss #total loss -> wiegthed by parameter P

                # P = 0: classification only.
                # P = 1: heatmaps only.
                # Intermediate values: trade-off between both.

                #print(f"Epoch {epoch}, Loss: {loss.item()}, cLoss: {cLoss.item()}, hLoss: {hLoss.item()}")

            else:#Common DenseNet, heat False
                
                loss=self._compute_loss(all_i, all_x, all_y, all_a, step) #only classification loss
        
        if(torch.isnan(loss).any()):
            raise ValueError('NaN Training Loss')

        loss.backward() #backpropagate to compute gradients

        if self.clip is not None: #gradient clipping, to avoid exploding gradients
            if self.clip!=0:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.clip)


        self.optimizer.step()#The optimizer updates the model's parameters using the computed gradients.

        if self.heat: #return losses
            return {
                "loss": loss.item(),
                "cLoss": cLoss.item(),
                "hLoss": hLoss.item()
            }
        else:
            return {"loss": loss.item()}
           
    def return_feats(self, x): 
        return self.network.extract_features(x) #retrun extacted features

    def predict(self, x):
        #x = x.to(next(self.network.parameters()).device)  #ensures the input is on the same device as the network
        #x.requires_grad_(True)  #enables gradient computation on the input
        out = self.network(x) #Forward pass
        
        return out 

    def returnBackbone(self):
        return self.network.returnBackbone()
   
    #----method Tune Cut 
    def initTuneCut(self,epochs):
        # train for self.cutEpochs to determine cut values; heatmap loss is not used
        self.tuneCut=True

        self.cutEpochs=int(epochs)
        self.aggregateE = {}
        self.keys = getattr(self, "keys", [])
        self.resetCut()
        self.heat = False

        # default temporary values 
        self.cut =  1e-5
        self.cut2 =  1000.0

        print("Initial cut values")
        print(f"cut: {self.cut}, cut2: {self.cut2}")
            
    def resetCut(self):
        self.aggregateE={}

        for name in getattr(self, "keys", []):
            self.aggregateE[name] = [0, 0, 0]

    def updateWelford(self,existingAggregate,newValue):
        (count, mean, M2) = existingAggregate
        count += 1
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2
        return (count, mean, M2)
    
    
    def updateCut(self,maps):
        if not hasattr(self, 'aggregateE'):
            self.resetCut()

        for layer in maps.keys():
            # If the layer is not yet in self.keys, add it
            if layer not in getattr(self, "keys", []):
                # add to keys and initialize aggregate
                self.keys = getattr(self, "keys", []) + [layer]
                self.aggregateE[layer] = [0, 0, 0]
        
        for layer in self.keys:

            if layer not in maps:
                # If the layer wasn't returned in maps this call, skip it.
                continue
            
            mapAbs=maps[layer]
            
            mapAbsZ=mapAbs[:,:int(mapAbs.shape[1]/2)]
            mapAbsE=mapAbs[:,int(mapAbs.shape[1]/2):]

            for i,_ in enumerate(mapAbsE,0):#batch iteration
                valueE=torch.mean(mapAbsE[i].detach().float()).item()
                
                if self.aggregateE.get(layer) is None:
                    self.aggregateE[layer] = [0, 0, 0]
                
                self.aggregateE[layer]=self.updateWelford(self.aggregateE[layer],valueE)

    def finalizeWelford(self,existingAggregate):
        # Retrieve the mean, variance and sample variance from an aggregate
        (count, mean, M2) = existingAggregate
        if count < 2:
            return float("nan")
        else:
            mean, sampleVariance = mean, M2 / (count - 1)
            std=sampleVariance**(0.5)
            return mean, std
        
    def returnCut(self):
        self.tuneCut=False #to continue with normal training 
        cut0={}
        cut1={}
        means={}
        stds={}
            
        for layer in self.keys:
            means[layer],stds[layer],cut0[layer],cut1[layer]=[],[],[],[]
            #order: Z, E, En
            
            mean,std=self.finalizeWelford(self.aggregateE[layer])
            means[layer].append(mean)
            stds[layer].append(std)
            c0=np.maximum(mean/5,mean-3*std)
            c1=np.minimum(c0*25,mean+3*std)
            cut0[layer].append(c0)
            cut1[layer].append(c1)

        return cut0,cut1,means,stds
        
#=========================ISNet===========================



#==============DOMAIN ADERSARIAL NEURAL NETWORK - DANN==========


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams,
                 grp_sizes=None, attr_sizes=None, conditional=False, class_balance=False):
        super(AbstractDANN, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes, attr_sizes)

        self.register_buffer('update_count', torch.tensor([0])) # counts update() calls; controls D/G alternation
        self.conditional = conditional # if True, discriminator is conditioned on class -> CDANN
        self.class_balance = class_balance  # if True, reweights discriminator loss by class frequency

        # network architecture
        self.featurizer = networks.Featurizer(data_type, input_shape, self.hparams) #extracts features
        self.classifier = networks.Classifier( #predicts class label from features
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier']
        )
        self.discriminator = networks.MLP(self.featurizer.n_outputs, num_attributes, self.hparams) #predicts demographic attribute
        self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        # optimizers -> dsic and generador -> SGD
        self.disc_opt = torch.optim.SGD(
            (list(self.discriminator.parameters()) + list(self.class_embeddings.parameters())), # +class_embeddings avoids having to write two separate optimizers
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            momentum=0.9)

        self.gen_opt = torch.optim.SGD(
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            momentum=0.9)

    def update(self, minibatch, step, current_epoch):
        all_i, all_x, all_y, all_a = minibatch
        self.update_count += 1 #increments update count
        all_z = self.featurizer(all_x) # extract features

        #--------------Discriminator-----------------
        if self.conditional: # CDANN: class-conditioned discriminator
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        
        disc_out = self.discriminator(disc_input) # predict attribute a

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, all_a, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, all_a) #loss del discriminador

        disc_softmax = F.softmax(disc_out, dim=1)

        # gradient penalty:
        # computes ∂p(attr_true)/∂disc_input and penalizes its squared norm.
        input_grad = autograd.grad(disc_softmax[:, all_a].sum(),
                                   [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)

        #print("[DEBUG] grad_penalty.shape:", grad_penalty.shape)

        #add gradient penalty to discriminator los
        disc_loss += self.hparams['grad_penalty'] * grad_penalty
        
        #training alternation:
        #trains D for d_steps_per_g steps per generator step
        d_steps_per_g = self.hparams['d_steps_per_g_step']
        
        if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:
            # DISCRIMINATOR (D) step
            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            # GENERATOR (G) step (G = featurizer + classifier)
            #preds del generator
            all_preds = self.classifier(all_z)

            classifier_loss = F.cross_entropy(all_preds, all_y) #normal classifier loss
            
            
            # generator loss:
            # minimize classification loss +
            # maximize discriminator loss (adversarial) - fools it using a disc_loss weighted by lambda

            gen_loss = classifier_loss + (self.hparams['lambda'] * -disc_loss)
            
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def return_feats(self, x):
        return self.featurizer(x)

    def predict(self, x):
        return self.classifier(self.featurizer(x))

class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None, attr_sizes=None):
        super(DANN, self).__init__(data_type, input_shape, num_classes, num_attributes, num_examples, hparams,
                                   grp_sizes, attr_sizes, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None, attr_sizes=None):
        super(CDANN, self).__init__(data_type, input_shape, num_classes, num_attributes, num_examples, hparams,
                                    grp_sizes, attr_sizes, conditional=True, class_balance=True)


#==============DOMAIN ADERSARIAL NEURAL NETWORK- DANN==========


#==============ISNetDANN====================

class ISNetDANN(Algorithm):
    """DANN with ISNet as generator"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, conditional=False, class_balance=False, nesterov=False, dropLr=None, val='ISNet', explainLabels=False, grp_sizes=None, attr_sizes=None):
        
        super(ISNetDANN, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes, attr_sizes)

        #---vars ISNet------
        self.hparams = hparams
        self.heat=self.hparams['heat'] #default: True
        self.lr=self.hparams['lr']
        self.optimizer=self.hparams['optimizer']
        self.momentum=self.hparams['momentum']
        self.WD=self.hparams['weight_decay']
        self.nesterov=nesterov 
        self.clip=self.hparams['clip']
        self.dropLr=dropLr 
        self.penalizeAll=self.hparams['penalizeAll']
        self.dLoss=self.hparams['dLoss']
        self.testLoss=False 
        self.alternativeForeground=self.hparams['alternativeForeground']
        self.norm=self.hparams['norm']
        self.val=val 
        #self.explainLabels=explainLabels

        self.Phparams=self.hparams['P']

        # checks whether P from hparams is a dictionary or a fixed value
        if isinstance(self.Phparams, dict): 
            self.P=self.Phparams[0] 
            self.increaseP=self.Phparams 

        else:
            self.P=self.Phparams 
            self.increaseP=None

        self.d=self.hparams['d']

        self.cut=self.hparams['cut']
        self.cut2=self.hparams['cut2']
        self.cutEpochs = self.hparams['tuneCutEpochs']
        #self.tuneCut=self.hparams['tuneCut']
        self.tuneCut = self.hparams.get('tuneCut', False)  
        self.A=self.hparams['A']
        self.B=self.hparams['B']
        self.E=self.hparams['Ea']
        #---vars ISNet------
        

        #---vars DANN-------
        # By default, DANN runs with conditional=False and class_balance=False
        self.register_buffer('update_count', torch.tensor([0])) # counts how many times update is called; used to alternate D vs G training
        self.conditional = conditional # if True, the discriminator also receives the predicted class → CDANN
        self.class_balance = class_balance # if True, rebalances the discriminator loss according to class distribution
        #---vars DANN-------
        

        #-----constructions of the networks----

        #ISNet
        self.isnet_network = networks.Complete_Net(input_shape, self.hparams)  # creation of the full network, including Featurizer and Classifier


        #update heatmap parameter
        self.isnet_network.heat = self.heat
        
        self.featurizer = self.isnet_network.extract_backbone_features() # extracts the backbone from the featurizer, to pass the image: feats = self.featurizer(x)

        
        # MLP -> discriminator
        self.discriminator = networks.MLP(self.isnet_network.n_outputs, num_attributes, self.hparams) #predicts the demographic attribute
        self.class_embeddings = nn.Embedding(num_classes, self.isnet_network.n_outputs) # in CDANN, added to the features to condition the discriminator on the class


        self._init_model()
        
    def _init_model(self):
        
        self.disc_opt = torch.optim.SGD(
            (list(self.discriminator.parameters()) + list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            momentum=0.9)

        self.gen_opt = torch.optim.SGD(
            list(self.isnet_network.parameters()), #use all ISNet parameters (both backbone and head)
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            momentum=0.9)

        self.lr_scheduler = None

        if self.dropLr is not None:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.gen_opt,
                milestones=self.dropLr,
                verbose=True
            )
        else:
            self.scheduler = None

    def _compute_loss_ISNet(self, i, x, y, a, step, masks=None, norm=None): 

        out = self.predict(x) #Forward pass through the complete net

        if norm is None:
            norm=self.norm

        # calls CompoundLoss from ISNetFlexTorch, which returns the losses
        loss = ISNetFlexTorch.CompoundLoss(out=out,
                                         labels=y,
                                         masks=masks,
                                         tuneCut=self.tuneCut, #Defalt: False
                                         d=self.d,
                                         dLoss=self.dLoss,
                                         cutFlex=self.cut,
                                         cut2Flex=self.cut2,
                                         A=self.A,
                                         B=self.B,
                                         E=self.E,
                                         alternativeForeground=self.alternativeForeground,
                                         norm=norm)


        if not self.heat or masks is None: #if heat is False or no masks are provided, return only the classification loss
            return loss['classification']
        if not self.tuneCut: #if heat is True, masks True, and tuneCut is False → standard ISNet with the two losses: heatmap loss and classification loss
            return loss['classification'],loss['LRPFlex']
        else:
            self.keys=list(loss['mapAbsFlex'].keys()) # if heat is True, masks exist, and tuneCut is True
            return loss['classification'],loss['LRPFlex'],loss['mapAbsFlex']

    
    def update(self, minibatch, step, current_epoch):
        """Perform one update step."""

        all_i, all_x, all_y, all_a, all_mask = minibatch

        #training_step
        self.step=step
        self.current_epoch= current_epoch
        
        self.update_count += 1 # updates the counter of update calls

        
        all_z = self.featurizer(all_x) # generates the features (feature vector)
        all_z = torch.flatten(all_z, 1)

        #--------------DISCRIMINATOR-----------------
        
        if self.conditional: # CDANN
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z

        disc_out = self.discriminator(disc_input) #discriminato predicts a (demographic attribute)

        disc_pred = torch.argmax(disc_out, dim=1) #discriminator predictions (most likely class)
        
        # accuracy = percentage of correct predictions
        disc_acc = (disc_pred == all_a).float().mean().item()

        
        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, all_a, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, all_a) #discriminator loss

        disc_softmax = F.softmax(disc_out, dim=1)

        # gradient penalty:
        # computes ∂p(attr_true)/∂disc_input and penalizes its squared norm.
        input_grad = autograd.grad(disc_softmax[:, all_a].sum(),
                                   [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)

        #print("[DEBUG] grad_penalty.shape:", grad_penalty.shape)

        #add gradient penalty to discriminator loss
        disc_loss += self.hparams['grad_penalty'] * grad_penalty
        
        #training alternation:
        #trains D for d_steps_per_g steps per generator step
        d_steps_per_g = self.hparams['d_steps_per_g_step']
        
        if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:
           # DISCRIMINATOR (D) step
            #print("disc step starts")
            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            #print("disc step ends")
            #return {'disc_loss': disc_loss.item()}
            return {'disc_loss': disc_loss.item(),
                    'disc_acc': disc_acc}
            
        else:
            #--------------GENERATOR--------------
            #-------ISNet #ENTRENAMIENTO NORMAL
            """
            if (self.increaseP is not None):
                epochs=list(self.increaseP.keys()) 
                epochs.sort()
                for epoch in epochs:
                    if (self.current_epoch>=epoch):
                        self.P=self.increaseP[epoch]
            """
            #parameters
            P_min = 0.05  #minimum value of P
            P_target = 0.40 #target value of P
            ramp_end_step = 15000  #step at which P reaches P_target (~15000 steps to reach target)
            ramp_start_step = 0  #step at which ramping starts
            
            # Determine current value of P based on the current step
            if self.step <= ramp_start_step:
                self.P = P_min # Before the ramp starts, P stays at the minimum value
            elif self.step >= ramp_end_step:
                self.P = P_target# After the ramp ends, P stays at the target value
            else:
                # During the ramp, interpolate linearly between P_min and P_target
                progress = (self.step - ramp_start_step) / (ramp_end_step - ramp_start_step)  # Progress ratio [0,1]
                # Optional smoothing using a cosine function for a smooth start/stop:
                # import math
                # progress_smooth = 0.5 * (1 - math.cos(math.pi * progress))
                # self.P = P_min + progress_smooth * (P_target - P_min)

                #Linear interpolation (default)
                self.P = P_min + progress * (P_target - P_min)


            if (self.heat):#ISNet # NORMAL Training
                
                #classification losss and heatmap loss
                cLoss,hLoss=self._compute_loss_ISNet(all_i, all_x, all_y, all_a, step,
                                               masks=all_mask)
                
                isnet_loss = (1-self.P)*cLoss+self.P*hLoss #ISNet loss -> wiegthed by parameter P

                # P = 0: classification only
                # P = 1: heatmaps only
                # Intermediate values: trade-off between both


            else:#Common DenseNet, heat False
    
                isnet_loss = self._compute_loss_ISNet(all_i, all_x, all_y, all_a, step) #only classification loss
                    
            if(torch.isnan(isnet_loss).any()):
                raise ValueError('NaN Training Loss')
            
           
            # -----Generator loss-----
            # minimize isnet loss (includes classification loss and heatmap loss)
            # maximize discriminator loss (adversarial) - fools it using a disc_loss weighted by lambda

           
            gen_loss = isnet_loss + (self.hparams['lambda'] * -disc_loss)
            
            # -----Generator loss-----
            
            #clear gradients and backpropagation
            self.disc_opt.zero_grad() 
            self.gen_opt.zero_grad()
            gen_loss.backward()

            
            if self.clip is not None: 
                if self.clip!=0:
                    torch.nn.utils.clip_grad_norm_(self.isnet_network.parameters(), max_norm=self.clip)
            

            self.gen_opt.step()
           
            #return metrics
            if self.heat:
                return {
                    "gen_loss": gen_loss.item(),
                    "isnet_loss": isnet_loss.item(),
                    "cLoss": cLoss.item(),
                    "hLoss": hLoss.item(),
                    "P": self.P 
                }
            else:
                return {
                    "gen_loss": gen_loss.item(),
                    "isnet_loss": isnet_loss.item()
                }
            
            

    def return_feats(self, x):
        features = self.featurizer(x)
        features = torch.flatten(features, 1)
        return features

    def predict(self, x):
        #out = self.network(x)
        out =self.isnet_network(x) #Forward pass through the isnet network
        return out 
        
#==============ISNetDANN====================

