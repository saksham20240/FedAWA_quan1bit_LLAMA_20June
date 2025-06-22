#!/usr/bin/env python3
"""
Medical Federated Learning Models Package
- Llama 7B (Server) and Llama 3B (Client) models
- Medical Q&A specialized architectures
- FedLAW support for parameter-efficient training
"""

# Import medical Llama models (only the ones that actually exist)
try:
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
    _llama_models_available = True
except ImportError as e:
    print(f"Warning: Could not import some Llama models: {e}")
    _llama_models_available = False

# Try to import legacy models from a separate module or create redirects
try:
    # If you have these models in a separate file, import them
    # from .legacy_models import (
    #     CNNCifar10, CNNCifar100, ResNet20, ResNet18, 
    #     MLP, LeNet5, CNNfmnist
    # )
    
    # For now, create redirects to medical models if legacy models don't exist
    if _llama_models_available:
        # Redirect legacy models to medical equivalents
        CNNCifar10 = MedicalLlama3B
        CNNCifar100 = MedicalLlama3B  
        ResNet20 = MedicalLlama3B
        ResNet18 = MedicalLlama3B
        MLP = MedicalLlama3B
        LeNet5 = MedicalLlama3B
        CNNfmnist = MedicalLlama3B
        
        # FedLAW versions
        CNNCifar10_fedlaw = MedicalLlama3B_fedlaw
        CNNCifar100_fedlaw = MedicalLlama3B_fedlaw
        ResNet20_fedlaw = MedicalLlama3B_fedlaw
        ResNet18_fedlaw = MedicalLlama3B_fedlaw
        MLP_fedlaw = MedicalLlama3B_fedlaw
        LeNet5_fedlaw = MedicalLlama3B_fedlaw
        CNNfmnist_fedlaw = MedicalLlama3B_fedlaw
        
        _legacy_models_available = True
    else:
        _legacy_models_available = False
        
except ImportError as e:
    print(f"Warning: Could not set up legacy model redirects: {e}")
    _legacy_models_available = False

# Try to import reparam function for FedLAW
try:
    from .reparam_function import ReparamModule
except ImportError:
    # Create a basic version if not available
    import torch.nn as nn
    
    class ReparamModule(nn.Module):
        """Basic ReparamModule for FedLAW compatibility"""
        def __init__(self):  # Fixed: was **init**
            super(ReparamModule, self).__init__()
        
        def get_params(self):
            """Get model parameters for FedLAW"""
            return [p for p in self.parameters() if p.requires_grad]
        
        def set_params(self, params):
            """Set model parameters for FedLAW"""
            current_params = self.get_params()
            for current_p, new_p in zip(current_params, params):
                current_p.data.copy_(new_p.data)

# Define what's available for import
_available_exports = ['ReparamModule']

if _llama_models_available:
    _available_exports.extend([
        'MedicalLlama7B',
        'MedicalLlama3B', 
        'MedicalLlama7B_fedlaw',
        'MedicalLlama3B_fedlaw',
        'Llama7B_medical',
        'Llama3B_medical',
        'Llama7B_medical_fedlaw',
        'Llama3B_medical_fedlaw',
        'CNNfmnist_medical',
        'CNNfmnist_fedlaw_medical'
    ])

if _legacy_models_available:
    _available_exports.extend([
        'CNNCifar10',
        'CNNCifar100',
        'ResNet20', 
        'ResNet18',
        'MLP',
        'LeNet5',
        'CNNfmnist',
        'CNNCifar10_fedlaw',
        'CNNCifar100_fedlaw', 
        'ResNet20_fedlaw',
        'ResNet18_fedlaw',
        'MLP_fedlaw',
        'LeNet5_fedlaw',
        'CNNfmnist_fedlaw'
    ])

__all__ = _available_exports  # Fixed: was **all**

# Package metadata
__version__ = '1.0.0'  # Fixed: was **version**
__author__ = 'Medical Federated Learning Team'  # Fixed: was **author**
__description__ = 'Medical Q&A models for federated learning with Llama 7B/3B'  # Fixed: was **description**

def get_available_models():
    """Get list of available medical models"""
    models = {
        'server_models': [],
        'client_models': [],
        'legacy_redirects': []
    }
    
    if _llama_models_available:
        models['server_models'] = [
            'MedicalLlama7B',
            'MedicalLlama7B_fedlaw', 
            'Llama7B_medical',
            'Llama7B_medical_fedlaw'
        ]
        models['client_models'] = [
            'MedicalLlama3B',
            'MedicalLlama3B_fedlaw',
            'Llama3B_medical', 
            'Llama3B_medical_fedlaw'
        ]
    
    if _legacy_models_available:
        models['legacy_redirects'] = [
            'CNNCifar10', 'CNNCifar100', 'ResNet20', 'ResNet18', 
            'MLP', 'LeNet5', 'CNNfmnist'
        ]
    
    return models

def print_model_info():
    """Print information about available models"""
    models = get_available_models()
    
    print("🦙 Medical Federated Learning Models")
    print("=" * 50)
    
    if models['server_models']:
        print("🌐 Server Models (Llama 7B):")
        for model in models['server_models']:
            print(f"   - {model}")
    else:
        print("🌐 Server Models: Not available")
    
    if models['client_models']:
        print("\n🏥 Client Models (Llama 3B):")
        for model in models['client_models']:
            print(f"   - {model}")
    else:
        print("\n🏥 Client Models: Not available")
    
    if models['legacy_redirects']:
        print("\n⚠️ Legacy Model Redirects:")
        for model in models['legacy_redirects']:
            print(f"   - {model} → MedicalLlama3B")
    else:
        print("\n⚠️ Legacy Model Redirects: Not available")
    
    print("=" * 50)

if __name__ == "__main__":  # Fixed: was **name**
    print_model_info()
