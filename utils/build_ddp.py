import os
import random
import time
from typing import Tuple, List, Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sam.build_sam import build_sam2
from model.model import OmniSAM, OmniSAM_adapt

from dataset.Cityscapes import Cityscapes13, Cityscapes19
from dataset.WildPASS import WildPASSforPL, WildPASSforAdaptation
from dataset.DensePASS import DensePASS13 
from dataset.SynPASS import SynPASS13
from dataset.StanfordPin import StanfordPin8DataSet
from utils.label_spaces import CITYSCAPES13_CLASSES, CITYSCAPES19_CLASSES
from utils.checkpoint_adapt import smart_load_checkpoint

DATASET_NAME2CLASSES_13 = {
    'Stanford2D3D': ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"],
    'Cityscapes': CITYSCAPES13_CLASSES,
    'SynPASS': CITYSCAPES13_CLASSES,
    'WildPASS': CITYSCAPES13_CLASSES,
}

DATASET_NAME2CLASSES_19 = {
    'Cityscapes': CITYSCAPES19_CLASSES,
}


def resolve_class_names(dataset_name: str, class_set: int):
    if class_set == 13:
        return DATASET_NAME2CLASSES_13[dataset_name]
    if class_set == 19:
        if dataset_name not in DATASET_NAME2CLASSES_19:
            raise ValueError(f"dataset={dataset_name} does not support class_set=19 in this repo")
        return DATASET_NAME2CLASSES_19[dataset_name]
    raise ValueError(f"Unsupported class_set={class_set}")

# backbone registry: backbone -> (checkpoint_path, yaml_cfg_path)
SAM_CFGS: Dict[str, Tuple[str, str]] = {
    "sam2_l": (
        "sam/checkpoints/sam2.1_hiera_large.pt",
        "/configs/sam2.1/sam2.1_hiera_l.yaml",
    ),
    "sam2_b+": (
        "sam/checkpoints/sam2.1_hiera_base_plus.pt",
        "/configs/sam2.1/sam2.1_hiera_b+.yaml",
    ),
    "sam2_s": (
        "sam/checkpoints/sam2.1_hiera_small.pt",
        "/configs/sam2.1/sam2.1_hiera_s.yaml",
    ),
    "sam2_t": (
        "sam/checkpoints/sam2.1_hiera_tiny.pt",
        "/configs/sam2.1/sam2.1_hiera_t.yaml",
    ),
}

def build_model(args, num_classes, num_maskmem, device, local_rank):
    sam2_checkpoint, model_cfg = SAM_CFGS[args.backbone]
    sam = build_sam2(model_cfg, sam2_checkpoint, device=device)
    model = OmniSAM(sam, num_classes=num_classes, num_maskmem=num_maskmem).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    return model

def load_checkpoint(model, ckpt_path):
    if ckpt_path is None or ckpt_path == '':
        return
    smart_load_checkpoint(model, ckpt_path)

class DatasetBuilder:
    def __init__(self, args):
        self.args = args
    
    def build_src(self):
        class_set = getattr(self.args, "class_set", 13)
        class_names = resolve_class_names(self.args.dataset, class_set)
        num_classes = len(class_names)

        if self.args.dataset == 'Stanford2D3D':
            root_train = os.path.join(self.args.data_root, 'Stanford2d3d_Seg')
            train_list = 'dataset/s2d3d_pin_list/train.txt'
            val_list = 'dataset/s2d3d_pin_list/val.txt'
            train_dataset = StanfordPin8DataSet(root_train, train_list, crop_size=(1024, 1024), set='train')
            val_dataset = StanfordPin8DataSet(root_train, val_list, crop_size=(1024, 1024), set='val')
        elif self.args.dataset == 'SynPASS':
            root_train = os.path.join(self.args.data_root, 'SynPASS')
            train_list = 'dataset/synpass_list/train.txt'
            val_list = 'dataset/synpass_list/val.txt'
            train_dataset = SynPASS13(root_train, train_list, crop_size=(2048, 1024), sliding_window=(self.args.patch_stride, self.args.image_size))
            val_dataset = SynPASS13(root_train, val_list, crop_size=(2048, 1024), sliding_window=(self.args.patch_stride, self.args.image_size))
        elif self.args.dataset == 'Cityscapes':
            root_train = os.path.join(self.args.data_root, 'cityscapes')
            train_list = 'dataset/cityscapes_list/train.txt'
            val_list = 'dataset/cityscapes_list/val.txt'
            if class_set == 19:
                train_dataset = Cityscapes19(root_train, train_list, crop_size=(2048, 1024), sliding_window=(self.args.patch_stride, self.args.image_size), set='train')
                val_dataset = Cityscapes19(root_train, val_list, crop_size=(2048, 1024), sliding_window=(self.args.patch_stride, self.args.image_size), set='val')
            else:
                train_dataset = Cityscapes13(root_train, train_list, crop_size=(2048, 1024), sliding_window=(self.args.patch_stride, self.args.image_size), set='train')
                val_dataset = Cityscapes13(root_train, val_list, crop_size=(2048, 1024), sliding_window=(self.args.patch_stride, self.args.image_size), set='val')
        else:
            raise ValueError(self.args.dataset)
        
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            num_workers=8,
            pin_memory=True,
            sampler=train_sampler,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            num_workers=4,
            pin_memory=True,
            sampler=val_sampler,
        )  
                                
        return train_loader, val_loader, train_sampler, val_sampler
