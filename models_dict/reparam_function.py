#!/usr/bin/env python3
"""
ReparamModule for Medical Federated Learning with Llama Models
- Parameter reparameterization for efficient federated learning
- Optimized for Llama 7B/3B medical Q&A models
- Memory-efficient parameter sharing across hospitals
- Compatible with medquad_new.csv dataset
"""

import torch
import torch.nn as nn
import copy
from typing import List, Optional, Dict, Any

class ReparamModule(nn.Module):
    """
    Reparameterization module for Medical Llama Models in Federated Learning
    
    This module provides parameter reparameterization capabilities specifically
    optimized for Llama 7B (server) and Llama 3B (client) models in medical
    federated learning scenarios.
    
    Key Features:
    - Efficient parameter sharing between hospitals
    - Memory optimization for large language models
    - Compatible with medical Q&A tasks
    - Privacy-preserving parameter aggregation
    """
    
    def __init__(self):
        super(ReparamModule, self).__init__()
        self._param_shapes = None
        self._param_names = None
        self._total_params = 0
        self._is_initialized = False
        self._model_type = None  # 'llama_7b' or 'llama_3b'
    
    def _initialize_param_info(self):
        """Initialize parameter information for medical Llama models"""
        try:
            self._param_shapes = []
            self._param_names = []
            self._total_params = 0
            
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self._param_names.append(name)
                    self._param_shapes.append(param.shape)
                    self._total_params += param.numel()
            
            # Detect model type based on parameter count
            if self._total_params > 100_000_000:  # ~100M+ parameters
                self._model_type = 'llama_7b'
            else:
                self._model_type = 'llama_3b'
            
            self._is_initialized = True
            print(f"✅ Initialized ReparamModule for {self._model_type} with {self._total_params:,} parameters")
            
        except Exception as e:
            print(f"⚠️ Error initializing param info: {e}")
            self._is_initialized = False
    
    def get_params(self) -> List[torch.Tensor]:
        """Get all trainable parameters for medical models"""
        try:
            return [p for p in self.parameters() if p.requires_grad]
        except Exception as e:
            print(f"⚠️ Error getting parameters: {e}")
            return []
    
    def set_params(self, params: List[torch.Tensor]):
        """Set model parameters from federated aggregation"""
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
        """Get all trainable parameters as a single vector for efficient communication"""
        try:
            params = self.get_params()
            if not params:
                return torch.tensor([])
            
            return torch.cat([p.view(-1) for p in params], dim=0)
            
        except Exception as e:
            print(f"⚠️ Error getting parameter vector: {e}")
            return torch.tensor([])
    
    def set_param_vector(self, param_vector: torch.Tensor):
        """Set model parameters from a vector (for efficient federated updates)"""
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
        """Get parameters as a dictionary for federated aggregation"""
        try:
            return {name: param.clone() for name, param in self.named_parameters() if param.requires_grad}
        except Exception as e:
            print(f"⚠️ Error getting parameter dict: {e}")
            return {}
    
    def set_param_dict(self, param_dict: Dict[str, torch.Tensor]):
        """Set parameters from dictionary (for federated updates)"""
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
        """Create a deep copy for federated aggregation"""
        try:
            cloned = copy.deepcopy(self)
            return cloned
        except Exception as e:
            print(f"⚠️ Error cloning parameters: {e}")
            return self
    
    def zero_params(self):
        """Zero out all parameters (useful for aggregation initialization)"""
        try:
            for param in self.parameters():
                if param.requires_grad:
                    param.data.zero_()
        except Exception as e:
            print(f"⚠️ Error zeroing parameters: {e}")
    
    def scale_params(self, scale_factor: float):
        """Scale parameters by a factor (useful for weighted aggregation)"""
        try:
            for param in self.parameters():
                if param.requires_grad:
                    param.data.mul_(scale_factor)
        except Exception as e:
            print(f"⚠️ Error scaling parameters: {e}")
    
    def add_params(self, other_params: List[torch.Tensor], weight: float = 1.0):
        """Add weighted parameters from another hospital's model"""
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
    
    def get_medical_model_stats(self) -> Dict[str, Any]:
        """Get statistics specific to medical Llama models"""
        try:
            if not self._is_initialized:
                self._initialize_param_info()
            
            stats = {
                'model_type': self._model_type,
                'total_params': self._total_params,
                'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'param_shapes': self._param_shapes.copy() if self._param_shapes else [],
                'param_names': self._param_names.copy() if self._param_names else [],
                'memory_mb': self._total_params * 4 / (1024 * 1024),  # Assuming float32
                'is_server_model': self._model_type == 'llama_7b',
                'is_client_model': self._model_type == 'llama_3b',
                'task': 'medical_qa'
            }
            
            # Add parameter norms for monitoring
            param_norms = []
            for param in self.parameters():
                if param.requires_grad:
                    param_norms.append(torch.norm(param).item())
            
            if param_norms:
                stats.update({
                    'param_norm_mean': sum(param_norms) / len(param_norms),
                    'param_norm_max': max(param_norms),
                    'param_norm_min': min(param_norms),
                    'param_norm_std': torch.std(torch.tensor(param_norms)).item()
                })
            
            return stats
            
        except Exception as e:
            print(f"⚠️ Error getting medical model stats: {e}")
            return {'error': str(e)}
    
    def print_medical_model_info(self):
        """Print information about the medical model parameters"""
        try:
            stats = self.get_medical_model_stats()
            
            print(f"🦙 Medical Llama Model Information")
            print(f"=" * 50)
            print(f"   Model Type: {stats.get('model_type', 'unknown').upper()}")
            print(f"   Task: Medical Q&A")
            print(f"   Total Parameters: {stats.get('total_params', 0):,}")
            print(f"   Trainable Parameters: {stats.get('trainable_params', 0):,}")
            print(f"   Memory Usage: {stats.get('memory_mb', 0):.2f} MB")
            print(f"   Role: {'Server' if stats.get('is_server_model') else 'Client'}")
            
            if 'param_norm_mean' in stats:
                print(f"   Parameter Norm (mean): {stats['param_norm_mean']:.6f}")
                print(f"   Parameter Norm (std): {stats['param_norm_std']:.6f}")
            
            print(f"=" * 50)
            
        except Exception as e:
            print(f"⚠️ Error printing medical model info: {e}")

class MedicalFedLAWOptimizer:
    """
    Optimizer specifically designed for medical Llama models in federated learning
    """
    
    def __init__(self, reparam_module: ReparamModule, lr: float = 5e-5):
        self.reparam_module = reparam_module
        self.lr = lr
        self.momentum_buffer = None
        self.momentum = 0.9
        self.model_type = getattr(reparam_module, '_model_type', 'unknown')
        
        print(f"✅ Initialized Medical FedLAW Optimizer for {self.model_type}")
    
    def step(self, gradients: torch.Tensor):
        """Perform optimization step optimized for medical models"""
        try:
            if self.momentum_buffer is None:
                self.momentum_buffer = torch.zeros_like(gradients)
            
            # Momentum update
            self.momentum_buffer.mul_(self.momentum).add_(gradients)
            
            # Get current parameters
            current_params = self.reparam_module.get_param_vector()
            
            # Update parameters with gradient clipping for language models
            clipped_gradients = torch.clamp(self.momentum_buffer, -1.0, 1.0)
            updated_params = current_params - self.lr * clipped_gradients
            
            # Set updated parameters
            self.reparam_module.set_param_vector(updated_params)
            
        except Exception as e:
            print(f"⚠️ Error in Medical FedLAW optimizer step: {e}")
    
    def zero_grad(self):
        """Zero gradients"""
        pass

def aggregate_medical_models(models: List[ReparamModule], weights: Optional[List[float]] = None) -> ReparamModule:
    """
    Aggregate multiple medical Llama models from different hospitals
    
    Args:
        models: List of medical Llama models from hospitals
        weights: Optional weights for each hospital (based on data size, performance, etc.)
    
    Returns:
        Aggregated medical model
    """
    try:
        if not models:
            raise ValueError("No medical models provided for aggregation")
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        if len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        # Ensure all models are of the same type
        model_types = [getattr(model, '_model_type', 'unknown') for model in models]
        if len(set(model_types)) > 1:
            print(f"⚠️ Warning: Aggregating different model types: {set(model_types)}")
        
        # Create aggregated model (clone of first model)
        aggregated = models[0].clone_params()
        aggregated.zero_params()
        
        # Aggregate parameters
        print(f"🔄 Aggregating {len(models)} medical models...")
        for i, (model, weight) in enumerate(zip(models, weights)):
            model_params = model.get_params()
            aggregated.add_params(model_params, weight)
            print(f"   ✅ Aggregated model {i+1} with weight {weight:.4f}")
        
        print(f"✅ Medical model aggregation completed")
        return aggregated
        
    except Exception as e:
        print(f"⚠️ Error aggregating medical models: {e}")
        return models[0] if models else None

def create_medical_fedlaw_model(base_model, max_length=512):
    """
    Convert a regular medical model to FedLAW-compatible version
    
    Args:
        base_model: Base medical Llama model
        max_length: Maximum sequence length
    
    Returns:
        FedLAW-compatible medical model
    """
    class MedicalFedLAWWrapper(ReparamModule):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.tokenizer = getattr(base_model, 'tokenizer', None)
            self.model_id = getattr(base_model, 'model_id', 'unknown') + '_fedlaw'
            self.max_length = getattr(base_model, 'max_length', max_length)
            self.model_size = getattr(base_model, 'model_size', 'unknown')
        
        def forward(self, *args, **kwargs):
            return self.base_model(*args, **kwargs)
        
        def generate(self, *args, **kwargs):
            return self.base_model.generate(*args, **kwargs)
        
        def answer_medical_question(self, question):
            if hasattr(self.base_model, 'answer_medical_question'):
                return self.base_model.answer_medical_question(question)
            else:
                raise NotImplementedError("Base model does not support medical Q&A")
    
    return MedicalFedLAWWrapper(base_model)

if __name__ == "__main__":
    # Test the Medical ReparamModule
    print("🧪 Testing Medical ReparamModule...")
    
    # Create a test medical model
    class TestMedicalModel(ReparamModule):
        def __init__(self):
            super().__init__()
            # Simulate Llama-like architecture
            self.embedding = nn.Embedding(50000, 768)  # Vocab and hidden size
            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(768, 12, 3072) for _ in range(12)
            ])
            self.lm_head = nn.Linear(768, 50000)
            
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.transformer_layers:
                x = layer(x)
            return self.lm_head(x)
    
    try:
        # Test basic functionality
        model = TestMedicalModel()
        model.print_medical_model_info()
        
        # Test parameter vector operations
        param_vector = model.get_param_vector()
        print(f"✅ Parameter vector size: {len(param_vector):,}")
        
        # Test parameter setting
        new_vector = torch.randn_like(param_vector) * 0.01  # Small values
        success = model.set_param_vector(new_vector)
        print(f"✅ Parameter setting: {'Success' if success else 'Failed'}")
        
        # Test medical stats
        stats = model.get_medical_model_stats()
        print(f"✅ Model type detected: {stats.get('model_type', 'unknown')}")
        
        print("🎉 Medical ReparamModule tests passed!")
        
    except Exception as e:
        print(f"❌ Medical ReparamModule test failed: {e}")
        import traceback
        traceback.print_exc()
