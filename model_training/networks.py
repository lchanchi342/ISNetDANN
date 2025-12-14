#networks.py
#neural networks architectures

#inlude:
#Featuizer and classfier or a method tu extract features
#Forward method: Forward pass through the nerwork
#train and freeze_bn: Overrides PyTorch's default train() behavior and Ensures BatchNorm layers remain frozen even in training mode (for stability)

import torch
import timm
import torch.nn as nn

#ISNet
from model_training.LRP import LRPDenseNetZe #DesneNet adapted for LRP
from model_training.LRP import ISNetFlexTorch
import torch.nn.functional as F


class PretrainedImageModel(torch.nn.Module):

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """Override the default train() to freeze the BN parameters."""
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules(): #iterates through all the model's layers
            if isinstance(m, nn.BatchNorm2d): #checks if the layer is BatchNorm2d
                m.eval() #sets the layer to evaluation mode (freezes its statistics). Only freezes batchnorm for stability

#Timmodel builds the DenseNet121 model -> used as a feature extractor
class TimmModel(PretrainedImageModel):

    def __init__(self, name, input_shape, hparams, pretrained=True, freeze_bn=False):
        super().__init__() # calls the constructor of the PretrainedImageModel class

        self.network = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self.n_outputs = self.network.num_features
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['last_layer_dropout'])

        if freeze_bn:
            self.freeze_bn()
        else:
            assert hparams['last_layer_dropout'] == 0.
            #Dropout is only allowed if freeze_bn=True.
            #If freeze_bn=False, BatchNorm continues updating its statistics, and Dropout
            #may introduce instability; therefore, Dropout must be set to 0


#For creating the DenseNet adapted for LRP
class DenseNetLRPZe(PretrainedImageModel):

    def __init__(self, input_shape, hparams, pretrained=True, freeze_bn=False,
                 model=None,architecture='densenet121',
                 classes=2,dropout=False, #se cambio classes a 2
                 HiddenLayerPenalization=False,
                 selective=False,selectiveEps=1e-7,Zb=True,e=0.01,
                 multiple=False,randomLogit=False,
                 VGGRemoveLastMaxpool=False,
                 explainLabels=False,
                 heat=True):
        
        super().__init__() # calls the constructor of the PretrainedImageModel class
        
        
        #Creation of the ISNet network based on ISNetFlexTorch.ISNetFlex
        self.network=ISNetFlexTorch.ISNetFlex(
            model=model,
            architecture=architecture,  
            e=e,
            classes=classes,
            dropout=dropout,
            HiddenLayerPenalization=HiddenLayerPenalization,
            selective=selective,        #Explains only the logit of the target class
            selectiveEps=selectiveEps,
            Zb=Zb,                      #propagation rule
            multiple=multiple,          #one LRP heatmap per sample if False
            randomLogit=randomLogit,
            pretrained=pretrained,
            VGGRemoveLastMaxpool=VGGRemoveLastMaxpool,
            explainLabels=explainLabels) #set explainLabels=False when defining ISNet

        self.heat=heat 

        self.n_outputs = self.network.num_features
        
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['last_layer_dropout'])

        if freeze_bn:
            self.freeze_bn()
        else:
            assert hparams['last_layer_dropout'] == 0.
            #Dropout is only allowed if freeze_bn=True.
            #If freeze_bn=False, BatchNorm continues updating its statistics, and Dropout
            #may introduce instability; therefore, Dropout must be set to 0

    
    def forward(self,x,labels=None):
        return self.network(x,runLRPFlex=self.heat,labels=labels)

    #feature extractor
    def extract_features(self,x):
        return self.network.return_features(x)

    def extract_backbone_features(self): #architecture
        return self.network.return_features_backbone()


#To create DenseNet121 -> the Featurizer is used
def Featurizer(data_type, input_shape, hparams):
    """Loads the DenseNet-121 model using the TimmModel class """
    if hparams['image_arch'] == 'densenet121' or hparams['image_arch'] == 'densenet_sup_in1k':
        return TimmModel('densenet121', input_shape, hparams, hparams['pretrained'])
    else:
        raise NotImplementedError(f"Arquitectura {hparams['image_arch']} no soportada.")


#clasiffier used in alagortithms.py en el ERM and DANN
def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear: 
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features) #default classifier - Linear 


# Creation of the DenseNet adapted for LRP for ISNet
def Complete_Net(input_shape, hparams): 
    """ returns the complete ISNet model ready for training """
    complete_net = DenseNetLRPZe(input_shape, hparams, 
                                    architecture=hparams['image_arch'], 
                                    e=float(hparams['e']),
                                    classes=2, #2 for binary prediction
                                    dropout=hparams['dropout'],
                                    HiddenLayerPenalization=hparams['penalizeAll'],
                                    selective=hparams['selective'],
                                    selectiveEps=float(hparams['selectiveEps']),
                                    Zb=hparams['Zb'],
                                    multiple=hparams['multiple'],
                                    randomLogit=hparams['random'],
                                    pretrained=hparams['pretrained'],
                                    heat=hparams['heat'])
    return complete_net



#for the discriminator -> DANN
class MLP(nn.Module):

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
                                      for _ in range(hparams['mlp_depth'] - 2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.input(x)
        x = self.dropout(x)
        #print("DEBUG forward: ", type(x), x.shape, x.dtype, x.device)

        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x
