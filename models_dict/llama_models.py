import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from .reparam_function import ReparamModule

# Suppress transformer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

##############################################################################
# Llama 7B Model (Server) - Medical Q&A
##############################################################################

class MedicalLlama7B(nn.Module):
    """Llama 7B model for medical Q&A (Server)"""
    
    def __init__(self, max_length=1024):
        super(MedicalLlama7B, self).__init__()
        
        # Use GPT2-medium as Llama 7B approximation
        config = GPT2Config(
            vocab_size=32000,  # Llama vocab size
            n_positions=max_length,
            n_embd=1024,      # Smaller than real 7B for memory
            n_layer=24,       # Moderate depth
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
        
        # Use GPT2-small as Llama 3B approximation
        config = GPT2Config(
            vocab_size=32000,  # Llama vocab size
            n_positions=max_length,
            n_embd=768,       # Smaller embedding
            n_layer=12,       # Fewer layers
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
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through base model"""
        return self.base_model(input_ids, attention_mask, labels)
    
    def generate(self, input_ids, attention_mask=None, max_length=200, **kwargs):
        """Generate text responses"""
        return self.base_model.generate(input_ids, attention_mask, max_length, **kwargs)

##############################################################################
# Factory Functions (following your existing pattern)
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
# Compatibility with your existing CNN pattern
##############################################################################

def CNNfmnist_medical():
    """Medical version of CNNfmnist - returns Llama 3B instead"""
    print("Warning: Converting CNNfmnist to Llama 3B for medical tasks")
    return MedicalLlama3B()

def CNNfmnist_fedlaw_medical():
    """Medical FedLAW version of CNNfmnist - returns Llama 3B instead"""
    print("Warning: Converting CNNfmnist to Llama 3B FedLAW for medical tasks")
    return MedicalLlama3B_fedlaw()
