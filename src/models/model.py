"""
Model module for the ConvNeXT image classification project.

This module provides functions to create and configure ConvNeXT models
for image classification, with support for different variants and
custom number of classes.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, List
import logging


class ConvNeXTClassifier(nn.Module):
    """
    A ConvNeXT-based classifier for image classification.
    
    This class wraps torchvision ConvNeXT models and provides
    a flexible interface for different ConvNeXT variants.
    """
    
    def __init__(
        self,
        convnext_variant: str = 'convnext_tiny',
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize the ConvNeXT classifier.
        
        Args:
            convnext_variant: ConvNeXT variant ('convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone layers
        """
        super(ConvNeXTClassifier, self).__init__()
        
        self.convnext_variant = convnext_variant
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.logger = logging.getLogger(__name__)
        
        # Load the appropriate ConvNeXT model
        self.model = self._load_convnext_variant(convnext_variant, pretrained)
        
        # Get the number of input features for the final layer
        num_features = self.model.classifier[2].in_features
        
        # Replace the final fully connected layer
        self.model.classifier[2] = nn.Linear(num_features, num_classes)
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        self.logger.info(f"Initialized {convnext_variant} with {num_classes} classes")
        self.logger.info(f"Pretrained: {pretrained}, Frozen backbone: {freeze_backbone}")
    
    def _load_convnext_variant(self, variant: str, pretrained: bool) -> nn.Module:
        """
        Load the specified ConvNeXT variant.
        
        Args:
            variant: ConvNeXT variant name
            pretrained: Whether to use pretrained weights
        
        Returns:
            ConvNeXT model
        """
        convnext_models = {
            'convnext_tiny': models.convnext_tiny,
            'convnext_small': models.convnext_small,
            'convnext_base': models.convnext_base,
            'convnext_large': models.convnext_large
        }
        
        if variant not in convnext_models:
            raise ValueError(
                f"Invalid ConvNeXT variant: {variant}. "
                f"Must be one of: {list(convnext_models.keys())}"
            )
        
        weights = 'DEFAULT' if pretrained else None
        model = convnext_models[variant](weights=weights)
        
        return model
    
    def _freeze_backbone(self):
        """Freeze the backbone layers of the model."""
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze the final classifier layer
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        self.logger.info("Backbone layers frozen, only final layer trainable")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers of the model."""
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.logger.info("All layers unfrozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)
    
    def get_model_info(self) -> Dict:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'convnext_variant': self.convnext_variant,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
        
        return info


def create_model(
    convnext_variant: str = 'convnext_tiny',
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> ConvNeXTClassifier:
    """
    Create a ConvNeXT classifier model.
    
    Args:
        convnext_variant: ConvNeXT variant ('convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze the backbone layers
    
    Returns:
        ConvNeXTClassifier instance
    """
    model = ConvNeXTClassifier(
        convnext_variant=convnext_variant,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    return model


def get_available_convnext_variants() -> List[str]:
    """
    Get list of available ConvNeXT variants.
    
    Returns:
        List of available ConvNeXT variant names
    """
    return ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large']


if __name__ == "__main__":
    # Test the model
    import torch
    
    # Create a model
    model = create_model(
        convnext_variant='convnext_tiny',
        num_classes=10,
        pretrained=False
    )
    
    # Print model info
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
    
    # Test with different variants
    print("\nAvailable ConvNeXT variants:")
    for variant in get_available_convnext_variants():
        print(f"  - {variant}")
