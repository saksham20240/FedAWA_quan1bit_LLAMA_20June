import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from .reparam_function import ReparamModule

# Suppress transformer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

##############################################################################
# Fixed Llama Models with Matching Dimensions
##############################################################################

class MedicalLlama7B(nn.Module):
    """Llama 7B model for medical Q&A (Server) - FIXED dimensions"""
    
    def __init__(self, max_length=1024):
        super(MedicalLlama7B, self).__init__()
        
        # Use consistent embedding size (1024) for both server and client
        config = GPT2Config(
            vocab_size=32000,  # Llama vocab size
            n_positions=max_length,
            n_embd=768,      # Keep 1024 for server
            n_layer=24,       # More layers for server
            n_head=12,
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
        """Generate text responses - FIXED duplicate parameters"""
        # Remove conflicting parameters
        generation_kwargs = {
            'input_ids': input_ids,
            'max_length': max_length,
            'do_sample': True,
            'temperature': 0.7,
            'num_beams': 2,  # Reduced to avoid conflicts
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'no_repeat_ngram_size': 2
        }
        
        if attention_mask is not None:
            generation_kwargs['attention_mask'] = attention_mask
        
        return self.model.generate(**generation_kwargs)

class MedicalLlama3B(nn.Module):
    """Llama 3B model for medical Q&A (Clients) - FIXED dimensions"""
    
    def __init__(self, max_length=1024):
        super(MedicalLlama3B, self).__init__()
        
        # Use SAME embedding size (1024) to match server for aggregation
        config = GPT2Config(
            vocab_size=32000,  # Llama vocab size
            n_positions=max_length,
            n_embd=768,      # FIXED: Use 1024 to match server
            n_layer=12,       # Fewer layers for clients
            n_head=12,        # Same number of heads
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
        """Generate text responses - FIXED duplicate parameters"""
        # Remove conflicting parameters
        generation_kwargs = {
            'input_ids': input_ids,
            'max_length': max_length,
            'do_sample': True,
            'temperature': 0.7,
            'num_beams': 2,  # Reduced to avoid conflicts
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'no_repeat_ngram_size': 2
        }
        
        if attention_mask is not None:
            generation_kwargs['attention_mask'] = attention_mask
            
        return self.model.generate(**generation_kwargs)

##############################################################################
# Alternative T5-based Models (like in your example)
##############################################################################

class MedicalT5_7B(nn.Module):
    """T5-based model for server (alternative to GPT2)"""
    
    def __init__(self, max_length=512):
        super(MedicalT5_7B, self).__init__()
        
        from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
        
        # T5-large configuration for server
        config = T5Config(
            vocab_size=32128,
            d_model=1024,
            d_kv=64,
            d_ff=4096,
            num_layers=24,
            num_heads=16,
            relative_attention_num_buckets=32,
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            initializer_factor=1.0,
            feed_forward_proj="relu",
            is_encoder_decoder=True,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=1,
        )
        
        self.model = T5ForConditionalGeneration(config)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')  # Use t5-small tokenizer
        self.model_id = 't5_7b_medical_server'
        self.max_length = max_length
        self.model_size = '7b'
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, attention_mask=None, max_length=200, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=2,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

class MedicalT5_3B(nn.Module):
    """T5-based model for clients (alternative to GPT2)"""
    
    def __init__(self, max_length=512):
        super(MedicalT5_3B, self).__init__()
        
        from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
        
        # T5-base configuration for clients  
        config = T5Config(
            vocab_size=32128,
            d_model=1024,      # SAME as server for aggregation compatibility
            d_kv=64,
            d_ff=2048,         # Smaller feed-forward
            num_layers=12,     # Fewer layers
            num_heads=16,
            relative_attention_num_buckets=32,
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            initializer_factor=1.0,
            feed_forward_proj="relu",
            is_encoder_decoder=True,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=1,
        )
        
        self.model = T5ForConditionalGeneration(config)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model_id = 't5_3b_medical_client'
        self.max_length = max_length
        self.model_size = '3b'
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, attention_mask=None, max_length=200, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=2,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

##############################################################################
# FedLAW-compatible Models with Fixed Dimensions
##############################################################################

class MedicalLlama7B_fedlaw(ReparamModule):
    """FIXED Llama 7B with ReparamModule support"""
    
    def __init__(self, max_length=1024):
        super(MedicalLlama7B_fedlaw, self).__init__()
        
        self.base_model = MedicalLlama7B(max_length)
        self.tokenizer = self.base_model.tokenizer
        self.model_id = 'llama_7b_medical_server_fedlaw'
        self.max_length = max_length
        self.model_size = '7b'
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.base_model(input_ids, attention_mask, labels)
    
    def generate(self, input_ids, attention_mask=None, max_length=200, **kwargs):
        return self.base_model.generate(input_ids, attention_mask, max_length, **kwargs)

class MedicalLlama3B_fedlaw(ReparamModule):
    """FIXED Llama 3B with ReparamModule support"""
    
    def __init__(self, max_length=1024):
        super(MedicalLlama3B_fedlaw, self).__init__()
        
        self.base_model = MedicalLlama3B(max_length)
        self.tokenizer = self.base_model.tokenizer
        self.model_id = 'llama_3b_medical_client_fedlaw'
        self.max_length = max_length
        self.model_size = '3b'
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.base_model(input_ids, attention_mask, labels)
    
    def generate(self, input_ids, attention_mask=None, max_length=200, **kwargs):
        return self.base_model.generate(input_ids, attention_mask, max_length, **kwargs)

##############################################################################
# Factory Functions
##############################################################################

def Llama7B_medical(max_length=1024):
    return MedicalLlama7B(max_length)

def Llama3B_medical(max_length=1024):
    return MedicalLlama3B(max_length)

def Llama7B_medical_fedlaw(max_length=1024):
    return MedicalLlama7B_fedlaw(max_length)

def Llama3B_medical_fedlaw(max_length=1024):
    return MedicalLlama3B_fedlaw(max_length)

# T5 alternatives
def T5_7B_medical(max_length=512):
    return MedicalT5_7B(max_length)

def T5_3B_medical(max_length=512):
    return MedicalT5_3B(max_length)

##############################################################################
# Compatibility Functions
##############################################################################

def CNNfmnist_medical():
    """Medical version - returns fixed Llama 3B"""
    print("Warning: Converting CNNfmnist to Llama 3B for medical tasks")
    return MedicalLlama3B()

def CNNfmnist_fedlaw_medical():
    """Medical FedLAW version - returns fixed Llama 3B"""
    print("Warning: Converting CNNfmnist to Llama 3B FedLAW for medical tasks")
    return MedicalLlama3B_fedlaw()
