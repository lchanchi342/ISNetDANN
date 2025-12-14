#datasets.py


import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Subset
from PIL import Image, ImageFile
from torchvision import transforms
import cv2

from model_training.augmentations import get_image_aug

from model_training.augmentations import JointTransformBasic2
from model_training.augmentations import JointTransformVal


DATASETS = ['MIMIC']
CXR_DATASETS = ['MIMIC']


ATTRS = ['sex', 'race', 'sex_race', 'age'] #demographic attributes
TASKS = ['No Finding', 'Pleural Effusion', 'Cardiomegaly', 'Pneumothorax'] #predeiction tasks -> pathologies


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError(f"Dataset not found: {dataset_name}")
    return globals()[dataset_name]


def num_environments(dataset_name): 
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class SubpopDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    INPUT_SHAPE = None       # Subclasses should override
    AVAILABLE_ATTRS = None   # Subclasses should override
    SPLITS = {               # Default, subclasses may override
        'tr': 0,
        'va': 1,
        'te': 2
    }
    EVAL_SPLITS = ['te']     # Default, subclasses may override

    def __init__(self, root, split, metadata, transform, group_def='group', subset_query=None, use_masks=False):
        if metadata is not None:
            df = pd.read_csv(metadata)
            df = df[df["split"] == (self.SPLITS[split])]

            if subset_query is not None:
                df = df.query(subset_query)
                if df['age'].min() == 1:
                    df['age'] = df['age'] - 1

            df['y'] = df[self.task_name]
            df['a'] = df[self.attr_name]
        
            self.idx = list(range(len(df)))
            
            self.x = df["path"].astype(str).map(lambda x: os.path.join(root, x)).tolist() #image paths
            self.y = df["y"].astype(int).tolist() #Labels of a selected pathology
            self.a = df["a"].astype(int).tolist() if group_def in ['group', 'attr'] else [0] * len(df["a"].tolist()) #demographic attribute label

            self.use_masks = use_masks
            
            if use_masks:
                required_columns = ['Left Lung', 'Right Lung', 'Heart']
                
                if all(col in df.columns and df[col].notna().any() for col in required_columns):
                    self.LL = df['Left Lung'].tolist()
                    self.RL = df['Right Lung'].tolist()
                    self.H = df['Heart'].tolist()
                    self.om_height = df['Height'].tolist()
                    self.om_width = df['Width'].tolist()
                
                else:
                    raise ValueError("No masks available: required columns are missing")
            
            self.transform_ = transform
            self._count_groups()


    def _count_groups(self): # Counts the number of samples in each demographic and class group.
        self.weights_g, self.weights_y, self.weights_a = [], [], []
        self.num_attributes = len(set(self.a))
        if self.num_attributes != max(set(self.a)) + 1:
            self.num_attributes = max(set(self.a)) + 1
        self.num_labels = len(set(self.y))
        self.group_sizes = [0] * self.num_attributes * self.num_labels
        self.class_sizes = [0] * self.num_labels
        self.attr_sizes = [0] * self.num_attributes

        for i in self.idx: # Counting samples in each category
            self.group_sizes[self.num_attributes * self.y[i] + self.a[i]] += 1
            self.class_sizes[self.y[i]] += 1
            self.attr_sizes[self.a[i]] += 1

        for i in self.idx: 
            self.weights_g.append(len(self) / self.group_sizes[self.num_attributes * self.y[i] + self.a[i]])
            self.weights_y.append(len(self) / self.class_sizes[self.y[i]])
            self.weights_a.append(len(self) / self.attr_sizes[self.a[i]])


    def __getitem__(self, index):
        i = self.idx[index]
        #x = self.transform(self.x[i])
        y = torch.tensor(self.y[i], dtype=torch.long)
        a = torch.tensor(self.a[i], dtype=torch.long)


        if self.use_masks:
            
            image = Image.open(self.x[i]).convert('RGB')
            
            om_height = self.om_height[i]
            om_width = self.om_width[i]

            #decode individual masks
            mask_LL = self.rle_decode(self.LL[i], (om_height, om_width))
            mask_RL = self.rle_decode(self.RL[i], (om_height, om_width))
            mask_H = self.rle_decode(self.H[i], (om_height, om_width))
            
            #combined masks
            combined_mask = np.clip(mask_LL + mask_RL + mask_H, 0, 1).astype(np.uint8)

            # ----dilation-----
            # Smooth general dilation
            kernel_general = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 30))
            dilated_general = cv2.dilate(combined_mask, kernel_general, iterations=1)
        
            # Apply smoothed expansion to the lungs -> downward
            expanded_left_base = self.expand_downward_smooth(mask_LL)
            expanded_right_base = self.expand_downward_smooth(mask_RL)
            
            # Final mask 
            final_dilated = np.clip(dilated_general + expanded_left_base + expanded_right_base, 0, 1).astype(np.uint8)
            # ----dilation-----

            # Convert mask to PIL image
            mask_img = Image.fromarray(final_dilated * 255).convert('L') # change here if dilation is not desired

            # Resize the mask so both have the same size and the transform can be applied
            mask_img  = mask_img .resize(image.size, resample=Image.LANCZOS)

            # Apply joint (simultaneous) transformation to img and mask
            x, mask = self.transform_(image, mask_img)

            return i, x, y, a, mask
            
        else:
            image = Image.open(self.x[i]).convert('RGB')
            x = self.transform_(image)
            #x = self.transform(self.x[i]) - original
            #mask = torch.zeros((1, x.shape[1], x.shape[2]))  

            return i, x, y, a
    

    def __len__(self):
        return len(self.idx)


    #------rle decode------------------
    @staticmethod
    def rle_decode(rle_str, shape):
        s = list(map(int, rle_str.split()))
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[::2], s[1::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)
    
    # ------------------ SMOOTH DOWNWARD EXPANSION ------------------
    #100 pixels - steps of 20 pixels
    @staticmethod
    def expand_downward_smooth(mask, max_shift=100, step=20):
        h, w = mask.shape
        expanded = np.zeros_like(mask, dtype=np.float32)
        
        for shift in range(0, max_shift + 1, step):
            shifted = np.roll(mask, shift, axis=0).astype(np.float32)
            shifted[:shift, :] = 0  
            decay = 1.0 - (shift / max_shift) 
            expanded += shifted * decay
    
        expanded = np.clip(expanded, 0, 1)
        
        expanded_blur = cv2.GaussianBlur(expanded, (21, 21), sigmaX=15, sigmaY=15)
        return (expanded_blur > 0.05).astype(np.uint8)
            


# BaseImageDataset -> defines the transform, loads the images, handles image datasets

class BaseImageDataset(SubpopDataset):

    def __init__(self, metadata, split, hparams, group_def='group', override_attr=None, subset_query=None, use_masks=False):
        
        if split == 'tr' and hparams['data_augmentation'] != 'none':
            #transform = get_image_aug(hparams['data_augmentation'], self.INPUT_SHAPE[1:], 256)

            #when masks are riquired -> ISNet - ISNetDANN
            if use_masks: # Apply joint (simultaneous) transformation to img and mask for training
                transform = JointTransformBasic2(target_shape=(256, 256))  
            else:
                transform = get_image_aug(hparams['data_augmentation'], self.INPUT_SHAPE[1:], 256)
        else:
            if use_masks: # Apply joint (simultaneous) transformation to img and mask for evaluation
                transform = JointTransformVal(target_size=256)  
            else:
                transform = transforms.Compose([
                transforms.Resize(256),  # Resizes the image to 256 pixels while keeping aspect ratio
                transforms.CenterCrop(256),  # Crops the center of the image to 256x256
                transforms.ToTensor(),   # Converts the image to a PyTorch tensor; scales values from [0,255] to [0,1]
                transforms.Normalize([0.5307, 0.5307, 0.5307], [0.2583, 0.2583, 0.2583]) #Normalize to mean and std values from MIMIC
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #Normalize to mean and std to ImageNet values
            ])
                
        
        self.data_type = "images"

        self.task_name = hparams['task']
       
        self.attr_name = override_attr if override_attr is not None else hparams['attr']

        # Calls the constructor of SubpopDataset (clase padre)
        super().__init__('/', split, metadata, transform, group_def, subset_query, use_masks)

    def transform(self, x): #x -> img path
        return self.transform_(Image.open(x).convert("RGB"))


# SubsetImageDataset -> creates a subset of an image dataset based on a list of indices.
class SubsetImageDataset(SubpopDataset):
    """
    Subsets from an existing dataset based on provided indices
    """
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)

    # ds: Original dataset.
    # idxs: List of indices to be selected from the original dataset.
    
    def __init__(self, ds, idxs):
        self.orig_ds = ds
        self.ds = Subset(ds, idxs) 
        self.x = [ds.x[i] for i in idxs]
        self.a = [ds.a[i] for i in idxs]
        self.y = [ds.y[i] for i in idxs]
        self.idx = list(range(len(self.x)))
        self.data_type = "images"
        self.attr_name = ds.attr_name
        self.task_name = ds.task_name
        self._count_groups()
        super().__init__(None, None, None, None) 

    def __getitem__(self, idx): 
        return self.ds[idx]

    def __len__(self):
        return len(self.ds)




class MIMIC(BaseImageDataset):
    N_STEPS = 30001 # number of iterations
    CHECKPOINT_FREQ = 1000 # model checkpoint frequency
    N_WORKERS = 16  # number of parallel processes for data loading
    INPUT_SHAPE = (3, 256, 256,) # input shape
    AVAILABLE_ATTRS = ['sex', 'race', 'sex_race', 'age'] # demo attrs
    TASKS = ['No Finding', 'Pleural Effusion', 'Cardiomegaly', 'Pneumothorax'] #pathologies


    def __init__(self, data_path, split, hparams, group_def='group', override_attr=None, subset_query=None,  use_masks=False):
        #metadata = Path("/home/lchanch/initial_training/image_df_mini")
        metadata = data_path
        super().__init__(metadata, split, hparams, group_def, override_attr, subset_query, use_masks)


