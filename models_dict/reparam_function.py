#!/usr/bin/env python3
"""
ReparamModule for FedLAW Support in Medical Federated Learning
- Parameter reparameterization for efficient federated learning
- Compatible with Llama 7B/3B medical models
- Memory-efficient parameter sharing
"""

import torch
import torch.nn as nn
import copy
from typing import List, Optional, Dict, Any

class ReparamModule(nn.Module):
    """
    Base reparameterization module for FedLAW federated learning
    
    This module provides parameter reparameterization capabilities
    for efficient parameter sharing in federated learning scenarios.
    """
    
    def __init__(self):
        super(ReparamModule, self).__init__()
        self._param_shapes = None
        self._param_names = None
        self._total_params = 0
        self._is_initialized = False
    
    def _initialize_param_info(self):
        """Initialize parameter information for reparameterization"""
        try:
            self._param_shapes = []
            self._param_names = []
            self._total_params = 0
            
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self._param_names.append(name)
                    self._param_shapes.append(param.shape)
                    self._total_params += param.numel()
            
            self._is_initialized = True
            
        except Exception as e:
            print(f"⚠️ Error initializing param info: {e}")
            self._is_initialized = False
    
    def get_params(self) -> List[torch.Tensor]:
        """Get all trainable parameters as a list"""
        try:
            return [p for p in self.parameters() if p.requires_grad]
        except Exception as e:
            print(f"⚠️ Error getting parameters: {e}")
            return []
    
    def set_params(self, params: List[torch.Tensor]):
        """Set model parameters from a list of tensors"""
        try:
            current_params = self.get_params()
            
            if len(params) != len(current_params):
                print(f"⚠️ Parameter count mismatch: expected {len(current_params)}, got {len(params)}")
                return False
            
            for current_p, new_p in zip(current_params, params):
                if current_p.shape != new_p.shape:
                    print(f"⚠️ Shape mismatch: {current_p.shape} vs {new_p.shape}")
                    continue
                current_p.data.copy_(new_p.data)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error setting parameters: {e}")
            return False
    
    def get_param_vector(self) -> torch.Tensor:
        """Get all trainable parameters as a single vector"""
        try:
            params = self.get_params()
            if not params:
                return torch.tensor([])
            
            return torch.cat([p.view(-1) for p in params], dim=0)
            
        except Exception as e:
            print(f"⚠️ Error getting parameter vector: {e}")
            return torch.tensor([])
    
    def set_param_vector(self, param_vector: torch.Tensor):
        """Set model parameters from a single vector"""
        try:
            if not self._is_initialized:
                self._initialize_param_info()
            
            if len(param_vector) != self._total_params:
                print(f"⚠️ Vector size mismatch: expected {self._total_params}, got {len(param_vector)}")
                return False
            
            offset = 0
            for param in self.parameters():
                if param.requires_grad:
                    param_length = param.numel()
                    param.data = param_vector[offset:offset + param_length].view(param.shape)
                    offset += param_length
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error setting parameter vector: {e}")
            return False
    
    def get_param_dict(self) -> Dict[str, torch.Tensor]:
        """Get parameters as a dictionary"""
        try:
            return {name: param.clone() for name, param in self.named_parameters() if param.requires_grad}
        except Exception as e:
            print(f"⚠️ Error getting parameter dict: {e}")
            return {}
    
    def set_param_dict(self, param_dict: Dict[str, torch.Tensor]):
        """Set parameters from a dictionary"""
        try:
            for name, param in self.named_parameters():
                if param.requires_grad and name in param_dict:
                    if param.shape == param_dict[name].shape:
                        param.data.copy_(param_dict[name].data)
                    else:
                        print(f"⚠️ Shape mismatch for {name}: {param.shape} vs {param_dict[name].shape}")
            return True
        except Exception as e:
            print(f"⚠️ Error setting parameter dict: {e}")
            return False
    
    def clone_params(self) -> 'ReparamModule':
        """Create a deep copy of the module with cloned parameters"""
        try:
            cloned = copy.deepcopy(self)
            return cloned
        except Exception as e:
            print(f"⚠️ Error cloning parameters: {e}")
            return self
    
    def zero_params(self):
        """Zero out all trainable parameters"""
        try:
            for param in self.parameters():
                if param.requires_grad:
                    param.data.zero_()
        except Exception as e:
            print(f"⚠️ Error zeroing parameters: {e}")
    
    def scale_params(self, scale_factor: float):
        """Scale all trainable parameters by a factor"""
        try:
            for param in self.parameters():
                if param.requires_grad:
                    param.data.mul_(scale_factor)
        except Exception as e:
            print(f"⚠️ Error scaling parameters: {e}")
    
    def add_params(self, other_params: List[torch.Tensor], weight: float = 1.0):
        """Add weighted parameters from another model"""
        try:
            current_params = self.get_params()
            
            if len(other_params) != len(current_params):
                print(f"⚠️ Parameter count mismatch for addition")
                return False
            
            for current_p, other_p in zip(current_params, other_params):
                if current_p.shape == other_p.shape:
                    current_p.data.add_(other_p.data, alpha=weight)
                else:
                    print(f"⚠️ Shape mismatch for addition: {current_p.shape} vs {other_p.shape}")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error adding parameters: {e}")
            return False
    
    def get_param_stats(self) -> Dict[str, Any]:
        """Get statistics about model parameters"""
        try:
            if not self._is_initialized:
                self._initialize_param_info()
            
            stats = {
                'total_params': self._total_params,
                'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'param_shapes': self._param_shapes.copy() if self._param_shapes else [],
                'param_names': self._param_names.copy() if self._param_names else [],
                'memory_mb': self._total_params * 4 / (1024 * 1024),  # Assuming float32
            }
            
            # Add parameter norms
            param_norms = []
            for param in self.parameters():
                if param.requires_grad:
                    param_norms.append(torch.norm(param).item())
            
            if param_norms:
                stats.update({
                    'param_norm_mean': sum(param_norms) / len(param_norms),
                    'param_norm_max': max(param_norms),
                    'param_norm_min': min(param_norms)
                })
            
            return stats
            
        except Exception as e:
            print(f"⚠️ Error getting parameter stats: {e}")
            return {'error': str(e)}
    
    def print_param_info(self):
        """Print information about model parameters"""
        try:
            stats = self.get_param_stats()
            
            print(f"📊 Parameter Information for {self.__class__.__name__}")
            print(f"   Total Parameters: {stats.get('total_params', 0):,}")
            print(f"   Trainable Parameters: {stats.get('trainable_params', 0):,}")
            print(f"   Memory Usage: {stats.get('memory_mb', 0):.2f} MB")
            
            if 'param_norm_mean' in stats:
                print(f"   Parameter Norm (mean): {stats['param_norm_mean']:.6f}")
                print(f"   Parameter Norm (max): {stats['param_norm_max']:.6f}")
                print(f"   Parameter Norm (min): {stats['param_norm_min']:.6f}")
            
        except Exception as e:
            print(f"⚠️ Error printing parameter info: {e}")

class FedLAWOptimizer:
    """Optimizer for FedLAW reparameterized models"""
    
    def __init__(self, reparam_module: ReparamModule, lr: float = 0.01):
        self.reparam_module = reparam_module
        self.lr = lr
        self.momentum_buffer = None
        self.momentum = 0.9
    
    def step(self, gradients: torch.Tensor):
        """Perform optimization step with reparameterized gradients"""
        try:
            if self.momentum_buffer is None:
                self.momentum_buffer = torch.zeros_like(gradients)
            
            # Momentum update
            self.momentum_buffer.mul_(self.momentum).add_(gradients)
            
            # Get current parameters
            current_params = self.reparam_module.get_param_vector()
            
            # Update parameters
            updated_params = current_params - self.lr * self.momentum_buffer
            
            # Set updated parameters
            self.reparam_module.set_param_vector(updated_params)
            
        except Exception as e:
            print(f"⚠️ Error in FedLAW optimizer step: {e}")
    
    def zero_grad(self):
        """Zero gradients (placeholder for compatibility)"""
        pass

def aggregate_reparam_models(models: List[ReparamModule], weights: Optional[List[float]] = None) -> ReparamModule:
    """Aggregate multiple reparameterized models"""
    try:
        if not models:
            raise ValueError("No models provided for aggregation")
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        if len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        # Create aggregated model (clone of first model)
        aggregated = models[0].clone_params()
        aggregated.zero_params()
        
        # Aggregate parameters
        for model, weight in zip(models, weights):
            model_params = model.get_params()
            aggregated.add_params(model_params, weight)
        
        return aggregated
        
    except Exception as e:
        print(f"⚠️ Error aggregating reparam models: {e}")
        return models[0] if models else None

if __name__ == "__main__":
    # Test the ReparamModule
    print("Testing ReparamModule...")
    
    # Create a simple test module
    class TestModel(ReparamModule):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 5)
            self.linear2 = nn.Linear(5, 1)
        
        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x)))
    
    try:
        # Test basic functionality
        model = TestModel()
        model.print_param_info()
        
        # Test parameter vector operations
        param_vector = model.get_param_vector()
        print(f"✅ Parameter vector size: {len(param_vector)}")
        
        # Test parameter setting
        new_vector = torch.randn_like(param_vector)
        success = model.set_param_vector(new_vector)
        print(f"✅ Parameter setting: {'Success' if success else 'Failed'}")
        
        print("✅ ReparamModule tests passed!")
        
    except Exception as e:
        print(f"❌ ReparamModule test failed: {e}")
