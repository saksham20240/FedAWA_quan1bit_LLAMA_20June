#!/usr/bin/env python3
"""
Medical Federated Learning Models Package
- Llama 7B (Server) and Llama 3B (Client) models
- Medical Q&A specialized architectures
- FedLAW support for parameter-efficient training
"""

# Import all medical Llama models
from .llama_models import (
    MedicalLlama7B, 
    MedicalLlama3B,
    MedicalLlama7B_fedlaw, 
    MedicalLlama3B_fedlaw,
    Llama7B_medical,
    Llama3B_medical,
    Llama7B_medical_fedlaw,
    Llama3B_medical_fedlaw,
    CNNfmnist_medical,
    CNNfmnist_fedlaw_medical
)

# Legacy model redirects (for backward compatibility)
from .llama_models import (
    CNNCifar10,
    CNNCifar100, 
    ResNet20,
    ResNet18,
    MLP,
    LeNet5,
    CNNfmnist,
    # FedLAW versions
    CNNCifar10_fedlaw,
    CNNCifar100_fedlaw,
    ResNet20_fedlaw, 
    ResNet18_fedlaw,
    MLP_fedlaw,
    LeNet5_fedlaw,
    CNNfmnist_fedlaw
)

# Optional: Try to import reparam function for FedLAW
try:
    from .reparam_function import ReparamModule
except ImportError:
    # Create a basic version if not available
    import torch.nn as nn
    
    class ReparamModule(nn.Module):
        """Basic ReparamModule for FedLAW compatibility"""
        def __init__(self):
            super(ReparamModule, self).__init__()
        
        def get_params(self):
            """Get model parameters for FedLAW"""
            return [p for p in self.parameters() if p.requires_grad]
        
        def set_params(self, params):
            """Set model parameters for FedLAW"""
            current_params = self.get_params()
            for current_p, new_p in zip(current_params, params):
                current_p.data.copy_(new_p.data)

__all__ = [
    # Medical Llama models
    'MedicalLlama7B',
    'MedicalLlama3B', 
    'MedicalLlama7B_fedlaw',
    'MedicalLlama3B_fedlaw',
    'Llama7B_medical',
    'Llama3B_medical',
    'Llama7B_medical_fedlaw',
    'Llama3B_medical_fedlaw',
    
    # Legacy redirects
    'CNNCifar10',
    'CNNCifar100',
    'ResNet20', 
    'ResNet18',
    'MLP',
    'LeNet5',
    'CNNfmnist',
    'CNNfmnist_medical',
    
    # FedLAW versions
    'CNNCifar10_fedlaw',
    'CNNCifar100_fedlaw', 
    'ResNet20_fedlaw',
    'ResNet18_fedlaw',
    'MLP_fedlaw',
    'LeNet5_fedlaw',
    'CNNfmnist_fedlaw',
    'CNNfmnist_fedlaw_medical',
    
    # Base classes
    'ReparamModule'
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'Medical Federated Learning Team'
__description__ = 'Medical Q&A models for federated learning with Llama 7B/3B'

def get_available_models():
    """Get list of available medical models"""
    return {
        'server_models': [
            'MedicalLlama7B',
            'MedicalLlama7B_fedlaw', 
            'Llama7B_medical',
            'Llama7B_medical_fedlaw'
        ],
        'client_models': [
            'MedicalLlama3B',
            'MedicalLlama3B_fedlaw',
            'Llama3B_medical', 
            'Llama3B_medical_fedlaw'
        ],
        'legacy_redirects': [
            'CNNCifar10', 'CNNCifar100', 'ResNet20', 'ResNet18', 
            'MLP', 'LeNet5', 'CNNfmnist'
        ]
    }

def print_model_info():
    """Print information about available models"""
    models = get_available_models()
    
    print("🦙 Medical Federated Learning Models")
    print("=" * 50)
    print("🌐 Server Models (Llama 7B):")
    for model in models['server_models']:
        print(f"   - {model}")
    
    print("\n🏥 Client Models (Llama 3B):")
    for model in models['client_models']:
        print(f"   - {model}")
    
    print("\n⚠️ Legacy Model Redirects:")
    for model in models['legacy_redirects']:
        print(f"   - {model} → MedicalLlama3B")
    
    print("=" * 50)

if __name__ == "__main__":
    print_model_info()
