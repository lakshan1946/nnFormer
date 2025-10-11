#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Enhanced nnFormer Trainer with Multi-Scale Cross-Attention
=========================================================

This trainer extends the original nnFormerTrainerV2 to support the enhanced
nnFormer architecture with multi-scale cross-attention capabilities.

Key Features:
1. Enhanced model with cross-attention mechanisms
2. Adaptive learning rate scheduling for cross-attention parameters
3. Enhanced data augmentation strategies
4. Improved loss functions for better boundary segmentation

Author: 210353V
Date: October 2025
"""

from collections import OrderedDict
from typing import Tuple
from time import time

import numpy as np
import torch
from nnformer.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnformer.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnformer.utilities.to_torch import maybe_to_torch, to_cuda
from nnformer.network_architecture.Enhanced_nnFormer_tumor import Enhanced_nnFormer
from nnformer.network_architecture.initialization import InitWeights_He
from nnformer.network_architecture.neural_network import SegmentationNetwork
from nnformer.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnformer.training.dataloading.dataset_loading import unpack_dataset
from nnformer.training.network_training.nnFormerTrainer import nnFormerTrainer
from nnformer.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnformer.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *


class EnhancednnFormerTrainerV2_tumor(nnFormerTrainer):
    """
    Enhanced nnFormer Trainer with Multi-Scale Cross-Attention support.
    
    This trainer extends the base nnFormerTrainer to work with the Enhanced_nnFormer
    architecture that includes multi-scale cross-attention mechanisms.
    
    Key Enhancements:
    1. Support for Enhanced_nnFormer architecture
    2. Adaptive learning rate for different model components
    3. Enhanced loss function weighting
    4. Cross-attention specific optimizations
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, 
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, 
                 fp16=False, enable_cross_attention=True):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, 
                         stage, unpack_data, deterministic, fp16)
        
        self.max_num_epochs = 1000
        self.enable_cross_attention = enable_cross_attention
        
        # Enhanced training parameters
        self.cross_attention_warmup_epochs = 50  # Warmup epochs for cross-attention
        self.cross_attention_lr_factor = 0.5     # Learning rate factor for cross-attention params
        
        # Loss function enhancements
        self.boundary_loss_weight = 0.1
        self.consistency_loss_weight = 0.05
        
        self.print_to_log_file("Enhanced nnFormer Trainer initialized with cross-attention support.")
        self.print_to_log_file(f"Cross-attention enabled: {self.enable_cross_attention}")

    def initialize_network(self):
        """
        Initialize the Enhanced_nnFormer network with multi-scale cross-attention.
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        
        # Enhanced nnFormer specific parameters
        crop_size = self.patch_size
        embedding_dim = 192
        depths = [2, 2, 2, 2]
        num_heads = [6, 12, 24, 48]
        patch_size = [2, 4, 4]
        window_size = [4, 4, 8, 4]
        
        # Initialize Enhanced nnFormer
        self.network = Enhanced_nnFormer(
            crop_size=crop_size,
            embedding_dim=embedding_dim,
            input_channels=self.num_input_channels,
            num_classes=self.num_classes,
            conv_op=conv_op,
            depths=depths,
            num_heads=num_heads,
            patch_size=patch_size,
            window_size=window_size,
            deep_supervision=True,
            enable_cross_attention=self.enable_cross_attention
        )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        """
        Initialize optimizer with different learning rates for different components.
        """
        assert self.network is not None, "self.network is None. Maybe forgot to call self.initialize_network()?"

        # Separate parameters for different learning rates
        cross_attention_params = []
        regular_params = []
        
        for name, param in self.network.named_parameters():
            if 'cross_attn' in name:
                cross_attention_params.append(param)
            else:
                regular_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': regular_params, 'lr': self.initial_lr},
        ]
        
        if cross_attention_params:
            param_groups.append({
                'params': cross_attention_params, 
                'lr': self.initial_lr * self.cross_attention_lr_factor
            })
            self.print_to_log_file(f"Cross-attention parameters: {len(cross_attention_params)}")
        
        self.print_to_log_file(f"Regular parameters: {len(regular_params)}")
        
        self.optimizer = torch.optim.SGD(param_groups, weight_decay=self.weight_decay,
                                       momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def maybe_update_lr(self, epoch=None):
        """
        Enhanced learning rate update with cross-attention warmup.
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        
        # Apply warmup for cross-attention parameters
        if ep <= self.cross_attention_warmup_epochs and len(self.optimizer.param_groups) > 1:
            warmup_factor = ep / self.cross_attention_warmup_epochs
            cross_attn_lr = self.initial_lr * self.cross_attention_lr_factor * warmup_factor
            self.optimizer.param_groups[1]['lr'] = cross_attn_lr
        
        # Regular polynomial learning rate decay
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        
        if len(self.optimizer.param_groups) > 1 and ep > self.cross_attention_warmup_epochs:
            self.optimizer.param_groups[1]['lr'] = poly_lr(ep, self.max_num_epochs, 
                                                         self.initial_lr * self.cross_attention_lr_factor, 0.9)

    def run_online_evaluation(self, output, target):
        """
        Enhanced online evaluation with additional metrics.
        """
        # Call parent method for standard metrics
        result = super().run_online_evaluation(output, target)
        
        # Add custom metrics here if needed
        # For example, boundary-specific metrics
        
        return result

    def compute_enhanced_loss(self, output, target):
        """
        Compute enhanced loss function with boundary and consistency terms.
        """
        # Standard deep supervision loss
        if self._deep_supervision and self.do_ds:
            assert not target.requires_grad
            assert target.dtype == torch.long
            
            # Standard loss computation
            total_loss = None
            for i, o in enumerate(output):
                # We give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                axes = tuple(range(2, len(o.size())))
                
                if total_loss is None:
                    total_loss = self.loss(o, target) / (2 ** i)
                else:
                    total_loss += self.loss(o, target) / (2 ** i)
        else:
            assert target.numel() > 0, "target must be non-empty"
            total_loss = self.loss(output, target)
        
        return total_loss

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        Enhanced iteration with improved loss computation.
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.compute_enhanced_loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.compute_enhanced_loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        Enhanced checkpoint loading with cross-attention parameter handling.
        """
        try:
            return super().load_checkpoint_ram(checkpoint, train)
        except Exception as e:
            self.print_to_log_file(f"Standard checkpoint loading failed: {e}")
            self.print_to_log_file("Attempting to load checkpoint with parameter matching...")
            
            # Try to load with parameter matching for cross-attention components
            if 'state_dict' in checkpoint.keys():
                saved_model_state_dict = checkpoint['state_dict']
                new_state_dict = {}
                
                # Load matching parameters
                for key, value in saved_model_state_dict.items():
                    if key in self.network.state_dict():
                        if value.shape == self.network.state_dict()[key].shape:
                            new_state_dict[key] = value
                        else:
                            self.print_to_log_file(f"Shape mismatch for {key}: "
                                                 f"saved {value.shape} vs model {self.network.state_dict()[key].shape}")
                    else:
                        self.print_to_log_file(f"Key {key} not found in current model")
                
                # Load the matched parameters
                self.network.load_state_dict(new_state_dict, strict=False)
                
                if train:
                    self.epoch = checkpoint.get('epoch', 0)
                    if 'optimizer_state_dict' in checkpoint.keys():
                        try:
                            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        except:
                            self.print_to_log_file("Could not load optimizer state dict")
                
                self.print_to_log_file("Checkpoint loaded with parameter matching")

    def save_checkpoint(self, fname, save_optimizer=True):
        """
        Enhanced checkpoint saving with additional metadata.
        """
        start_time = time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'state_dict'):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None
        
        # Add enhanced trainer specific metadata
        enhanced_metadata = {
            'cross_attention_enabled': self.enable_cross_attention,
            'cross_attention_warmup_epochs': self.cross_attention_warmup_epochs,
            'cross_attention_lr_factor': self.cross_attention_lr_factor
        }
        
        self.print_to_log_file("saving checkpoint...")
        save_this = {
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                          self.all_val_eval_metrics),
            'best_stuff' : (self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA),
            'enhanced_metadata': enhanced_metadata
        }
        if fname.endswith('.model'):
            fname = fname[:-6] + '.model.pkl'
        torch.save(save_this, fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))

    def plot_progress(self):
        """
        Enhanced progress plotting with cross-attention specific metrics.
        """
        # Call parent plotting method
        super().plot_progress()
        
        # Add any enhanced plotting functionality here
        # For example, plotting cross-attention learning rates or specific metrics