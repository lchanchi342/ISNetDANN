#augmentatiosn.py

#transformation applied to CXR images during training -> get_image_aug -> basic2
#transformations applied jointly to CXR images and segmentation masks during training -> JointTransformBasic2
#transformations applied jointly to CXR images and segmentation masks during evaluation (val/test sets) -> JointTransformVal

from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import torch


def get_image_aug(aug_type, target_shape, scale_size):
    if aug_type == "basic":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(target_shape, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif aug_type == 'basic2':
        transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(target_shape, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.ToTensor(),
            transforms.Normalize([0.5307, 0.5307, 0.5307], [0.2583, 0.2583, 0.2583]), # MIMIC mean and std values
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # ImageNet mean and std values
        ])
    elif aug_type == "auto_aug":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(target_shape, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif aug_type == "rand_aug":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(target_shape, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif aug_type == "trivial_aug":
        transform = transforms.Compose([
            transforms.Resize(scale_size),
            transforms.CenterCrop(target_shape[0]),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif aug_type == "augmix":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(target_shape, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.AugMix(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        raise NotImplementedError(f"Augmentation type [{aug_type}] not supported.")
    return transform


class JointTransformBasic2:
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def __call__(self, image, mask):
        
        #random rotation
        angle = random.uniform(-15, 15)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        #random horizontal flip 
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        #random resized crop
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.7, 1.0), ratio=(0.75, 1.33)
        )
        image = TF.resized_crop(image, i, j, h, w, self.target_shape, interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resized_crop(mask,  i, j, h, w, self.target_shape, interpolation=TF.InterpolationMode.NEAREST)

        # tensor conversion and image normalization
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.5307]*3, std=[0.2583]*3) # MIMIC mean and std values. #NOTE: Do not normalize if just testing image saving
        #image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.485, 0.456, 0.406]) # ImageNet mean and std values

        #Mask to binary tensor
        mask = (TF.to_tensor(mask) > 0.5).float()
        
        return image, mask

from torchvision.transforms import functional as TF

class JointTransformVal: #for evaluation, applied to images on val/test set
    def __init__(self, target_size=256):
        self.target_size = target_size

    def __call__(self, image, mask):
        # Resize
        image = TF.resize(image, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  self.target_size, interpolation=TF.InterpolationMode.NEAREST)

        # Center crop
        image = TF.center_crop(image, self.target_size)
        mask  = TF.center_crop(mask,  self.target_size)

        # To tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.5307]*3, std=[0.2583]*3) #MIMIC
        #image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.485, 0.456, 0.406]) #ImageNet

        mask = (TF.to_tensor(mask) > 0.5).float()  # binaria
        
        return image, mask
