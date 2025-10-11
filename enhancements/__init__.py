"""
Enhanced nnFormer Project
=======================

This package contains enhancements to the nnFormer 3D transformer architecture
for improved medical image segmentation performance.
"""

__version__ = "1.0.0"
__author__ = "210353V"
__email__ = "student@university.edu"

from . import attention
from . import evaluation
from . import experiments

__all__ = ['attention', 'evaluation', 'experiments']