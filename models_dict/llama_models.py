import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

# Suppress transformer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Optional import for ReparamModule
try:
    from .reparam_function import ReparamModule
except ImportError:
    # Create a basic ReparamModule if not available
    class ReparamModule(nn.Module):
        def __init__(self):
            super(ReparamModule, self).__init__()
        
        def get_params(self):
            return [p for p in self.parameters() if p.requires_grad]
        
        def set_params(self, params):
            current_params = self.get_params()
            for current_p, new_p in zip(current_params, params):
                current_p.data.copy_(new_p.data)

##############################################################################
# Llama 7B Model (Server) - Medical Q&A
##############################################################################

class MedicalLlama7B(nn.Module):
    """Llama 7B model for medical Q&A (Server)"""
    
    def __init__(self, max_length=1024):
        super(MedicalLlama7B, self).__init__()
        
        # Use GPT2-medium as Llama 7B approximation for memory efficiency
        config = GPT2Config(
            vocab_size=50257,  # GPT-2 vocab size (compatible)
            n_positions=max_length,
            n_embd=1024,      # Larger embedding for server
            n_layer=24,       # More layers for server
            n_head=16,
            activation_function='gelu_new',
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            use_cache=True
        )
        
        self.model = GPT2LMHeadModel(config)
        self.model_id = 'llama_7b_medical_server'
        self.max_length = max_length
        self.model_size = '7b'
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # Add medical special tokens
        self._add_medical_tokens()
        
        print(f"✅ {self.model_id} initialized - Server model ready")
    
    def _add_medical_tokens(self):
        """Add medical-specific special tokens"""
        special_tokens = {
            "additional_special_tokens": [
                "[MEDICAL_Q]", "[MEDICAL_A]", 
                "[PATIENT]", "[DOCTOR]",
                "[SYMPTOM]", "[TREATMENT]"
            ]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            print(f"   Added {num_added} medical tokens")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass for causal language modeling"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, attention_mask=None, max_length=200, **kwargs):
        """Generate text responses"""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            num_beams=4,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )

##############################################################################
# Llama 3B Model (Client) - Medical Q&A  
##############################################################################

class MedicalLlama3B(nn.Module):
    """Llama 3B model for medical Q&A (Clients)"""
    
    def __init__(self, max_length=1024):
        super(MedicalLlama3B, self).__init__()
        
        # Use GPT2-small as Llama 3B approximation for clients
        config = GPT2Config(
            vocab_size=50257,  # GPT-2 vocab size (compatible)
            n_positions=max_length,
            n_embd=768,       # Smaller embedding for clients
            n_layer=12,       # Fewer layers for clients
            n_head=12,
            activation_function='gelu_new',
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            use_cache=True
        )
        
        self.model = GPT2LMHeadModel(config)
        self.model_id = 'llama_3b_medical_client'
        self.max_length = max_length
        self.model_size = '3b'
        
        # Initialize tokenizer  
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # Add medical special tokens
        self._add_medical_tokens()
        
        print(f"✅ {self.model_id} initialized - Client model ready")
    
    def _add_medical_tokens(self):
        """Add medical-specific special tokens"""
        special_tokens = {
            "additional_special_tokens": [
                "[MEDICAL_Q]", "[MEDICAL_A]", 
                "[PATIENT]", "[DOCTOR]",
                "[SYMPTOM]", "[TREATMENT]"
            ]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            print(f"   Added {num_added} medical tokens")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass for causal language modeling"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, attention_mask=None, max_length=200, **kwargs):
        """Generate text responses"""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            num_beams=2,  # Smaller beams for efficiency
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )

##############################################################################
# FedLAW-compatible Llama Models
##############################################################################

class MedicalLlama7B_fedlaw(ReparamModule):
    """Llama 7B with ReparamModule support for FedLAW (Server)"""
    
    def __init__(self, max_length=1024):
        super(MedicalLlama7B_fedlaw, self).__init__()
        
        # Initialize base model
        self.base_model = MedicalLlama7B(max_length)
        self.tokenizer = self.base_model.tokenizer
        self.model_id = 'llama_7b_medical_server_fedlaw'
        self.max_length = max_length
        self.model_size = '7b'
        
        print(f"✅ {self.model_id} initialized - FedLAW Server model ready")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through base model"""
        return self.base_model(input_ids, attention_mask, labels)
    
    def generate(self, input_ids, attention_mask=None, max_length=200, **kwargs):
        """Generate text responses"""
        return self.base_model.generate(input_ids, attention_mask, max_length, **kwargs)

class MedicalLlama3B_fedlaw(ReparamModule):
    """Llama 3B with ReparamModule support for FedLAW (Clients)"""
    
    def __init__(self, max_length=1024):
        super(MedicalLlama3B_fedlaw, self).__init__()
        
        # Initialize base model
        self.base_model = MedicalLlama3B(max_length)
        self.tokenizer = self.base_model.tokenizer
        self.model_id = 'llama_3b_medical_client_fedlaw'
        self.max_length = max_length
        self.model_size = '3b'
        
        print(f"✅ {self.model_id} initialized - FedLAW Client model ready")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through base model"""
        return self.base_model(input_ids, attention_mask, labels)
    
    def generate(self, input_ids, attention_mask=None, max_length=200, **kwargs):
        """Generate text responses"""
        return self.base_model.generate(input_ids, attention_mask, max_length, **kwargs)

##############################################################################
# Factory Functions
##############################################################################

def Llama7B_medical(max_length=1024):
    """Create Llama 7B for server"""
    return MedicalLlama7B(max_length)

def Llama3B_medical(max_length=1024):
    """Create Llama 3B for client"""
    return MedicalLlama3B(max_length)

def Llama7B_medical_fedlaw(max_length=1024):
    """Create Llama 7B with FedLAW support for server"""
    return MedicalLlama7B_fedlaw(max_length)

def Llama3B_medical_fedlaw(max_length=1024):
    """Create Llama 3B with FedLAW support for client"""
    return MedicalLlama3B_fedlaw(max_length)

##############################################################################
# Compatibility with existing CNN pattern
##############################################################################

def CNNfmnist_medical():
    """Medical version of CNNfmnist - returns Llama 3B instead"""
    print("⚠️ Converting CNNfmnist to Llama 3B for medical tasks")
    return MedicalLlama3B()

def CNNfmnist_fedlaw_medical():
    """Medical FedLAW version of CNNfmnist - returns Llama 3B instead"""
    print("⚠️ Converting CNNfmnist to Llama 3B FedLAW for medical tasks")
    return MedicalLlama3B_fedlaw()

##############################################################################
# Additional Legacy Support
##############################################################################

# Map old model names to new medical models
def CNNCifar10():
    """Legacy support - redirect to medical model"""
    print("⚠️ Redirecting CNNCifar10 to MedicalLlama3B for medical tasks")
    return MedicalLlama3B()

def CNNCifar100():
    """Legacy support - redirect to medical model"""
    print("⚠️ Redirecting CNNCifar100 to MedicalLlama3B for medical tasks")
    return MedicalLlama3B()

def ResNet20(num_classes=10):
    """Legacy support - redirect to medical model"""
    print("⚠️ Redirecting ResNet20 to MedicalLlama3B for medical tasks")
    return MedicalLlama3B()

def ResNet18(num_classes=10):
    """Legacy support - redirect to medical model"""
    print("⚠️ Redirecting ResNet18 to MedicalLlama3B for medical tasks")
    return MedicalLlama3B()

def MLP():
    """Legacy support - redirect to medical model"""
    print("⚠️ Redirecting MLP to MedicalLlama3B for medical tasks")
    return MedicalLlama3B()

def LeNet5():
    """Legacy support - redirect to medical model"""
    print("⚠️ Redirecting LeNet5 to MedicalLlama3B for medical tasks")
    return MedicalLlama3B()

# FedLAW versions
def CNNCifar10_fedlaw():
    return MedicalLlama3B_fedlaw()

def CNNCifar100_fedlaw():
    return MedicalLlama3B_fedlaw()

def ResNet20_fedlaw(num_classes=10):
    return MedicalLlama3B_fedlaw()

def ResNet18_fedlaw(num_classes=10):
    return MedicalLlama3B_fedlaw()

def MLP_fedlaw():
    return MedicalLlama3B_fedlaw()

def LeNet5_fedlaw():
    return MedicalLlama3B_fedlaw()

def CNNfmnist():
    return MedicalLlama3B()

def CNNfmnist_fedlaw():
    return MedicalLlama3B_fedlaw()

if __name__ == "__main__":
    # Test the medical models
    print("Testing Medical Llama Models...")
    
    try:
        # Test server model
        print("\n🦙 Testing Server Model (Llama 7B):")
        server_model = MedicalLlama7B(max_length=512)
        print(f"   Model ID: {server_model.model_id}")
        print(f"   Vocab Size: {len(server_model.tokenizer)}")
        
        # Test client model
        print("\n🦙 Testing Client Model (Llama 3B):")
        client_model = MedicalLlama3B(max_length=512)
        print(f"   Model ID: {client_model.model_id}")
        print(f"   Vocab Size: {len(client_model.tokenizer)}")
        
        # Test FedLAW models
        print("\n🦙 Testing FedLAW Models:")
        server_fedlaw = MedicalLlama7B_fedlaw(max_length=512)
        client_fedlaw = MedicalLlama3B_fedlaw(max_length=512)
        print(f"   FedLAW Server: {server_fedlaw.model_id}")
        print(f"   FedLAW Client: {client_fedlaw.model_id}")
        
        # Test a simple forward pass
        print("\n🧪 Testing Forward Pass:")
        dummy_input = torch.randint(0, 1000, (1, 10))
        dummy_mask = torch.ones(1, 10)
        
        with torch.no_grad():
            server_output = server_model(dummy_input, dummy_mask)
            client_output = client_model(dummy_input, dummy_mask)
        
        print(f"   Server output shape: {server_output.logits.shape}")
        print(f"   Client output shape: {client_output.logits.shape}")
        
        print("\n✅ All medical model tests passed!")
        
    except Exception as e:
        print(f"\n❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
