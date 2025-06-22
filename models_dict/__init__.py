#!/usr/bin/env python3
"""
Medical Federated Learning Models Package
- Llama 7B (Server) and Llama 3B (Client) models only
- Medical Q&A specialized architectures
- FedLAW support for parameter-efficient training
- Dataset: Medical Q&A (question, answer) pairs only
"""

# Import medical Llama models only
try:
    from .llama_models import (
        MedicalLlama7B, 
        MedicalLlama3B,
        MedicalLlama7B_fedlaw, 
        MedicalLlama3B_fedlaw,
        Llama7B_medical,
        Llama3B_medical,
        Llama7B_medical_fedlaw,
        Llama3B_medical_fedlaw
    )
    _llama_models_available = True
    print("✅ Medical Llama models loaded successfully")
except ImportError as e:
    print(f"❌ Error: Could not import Llama models: {e}")
    _llama_models_available = False

# Import reparam function for FedLAW
try:
    from .reparam_function import ReparamModule
    print("✅ ReparamModule for FedLAW loaded successfully")
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
    
    print("⚠️ Using basic ReparamModule implementation")

# Define available exports (only Llama models)
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
        'Llama3B_medical_fedlaw'
    ])

__all__ = _available_exports

# Package metadata
__version__ = '2.0.0'
__author__ = 'Medical Federated Learning Team'
__description__ = 'Medical Q&A models for federated learning with Llama 7B/3B only'

def get_available_models():
    """Get list of available medical Llama models"""
    models = {
        'server_models': [],
        'client_models': []
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
    
    return models

def print_model_info():
    """Print information about available medical models"""
    models = get_available_models()
    
    print("🦙 Medical Federated Learning Models")
    print("=" * 60)
    print("📊 Dataset: Medical Q&A (question → answer)")
    print("🏥 Task: Medical Question Answering")
    print("🔄 Learning: Federated Learning across Hospitals")
    print("=" * 60)
    
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
    
    print("\n📋 Supported Features:")
    print("   ✅ Medical Question Answering")
    print("   ✅ Federated Learning")
    print("   ✅ FedLAW Parameter Efficiency")
    print("   ✅ Hospital Privacy Preservation")
    print("=" * 60)

def create_medical_qa_model(model_type='client', use_fedlaw=False, max_length=512):
    """
    Create a medical Q&A model
    
    Args:
        model_type: 'server' or 'client'
        use_fedlaw: Whether to use FedLAW version
        max_length: Maximum sequence length
    
    Returns:
        Medical Llama model instance
    """
    
    if not _llama_models_available:
        raise ImportError("Llama models are not available. Please check your installation.")
    
    if model_type == 'server':
        if use_fedlaw:
            return MedicalLlama7B_fedlaw(max_length)
        else:
            return MedicalLlama7B(max_length)
    elif model_type == 'client':
        if use_fedlaw:
            return MedicalLlama3B_fedlaw(max_length)
        else:
            return MedicalLlama3B(max_length)
    else:
        raise ValueError("model_type must be 'server' or 'client'")

def get_model_info():
    """Get information about the medical federated learning setup"""
    return {
        'supported_datasets': ['Medical Q&A'],
        'supported_models': ['Llama 7B (Server)', 'Llama 3B (Client)'],
        'task': 'Medical Question Answering',
        'learning_type': 'Federated Learning',
        'privacy_preserving': True,
        'fedlaw_support': True,
        'hospital_specialization': True
    }

if __name__ == "__main__":
    print_model_info()
    
    # Test model creation
    try:
        print("\n🧪 Testing Model Creation:")
        
        # Test server model
        server_model = create_medical_qa_model('server', use_fedlaw=False)
        print(f"✅ Server model created: {server_model.model_id}")
        
        # Test client model
        client_model = create_medical_qa_model('client', use_fedlaw=False)
        print(f"✅ Client model created: {client_model.model_id}")
        
        # Test FedLAW models
        server_fedlaw = create_medical_qa_model('server', use_fedlaw=True)
        print(f"✅ Server FedLAW model created: {server_fedlaw.model_id}")
        
        client_fedlaw = create_medical_qa_model('client', use_fedlaw=True)
        print(f"✅ Client FedLAW model created: {client_fedlaw.model_id}")
        
        print("\n🎉 All medical models working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error testing models: {e}")
