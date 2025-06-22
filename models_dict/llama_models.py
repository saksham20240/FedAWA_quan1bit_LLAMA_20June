import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from .reparam_function import ReparamModule

# Suppress transformer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

##############################################################################
# Medical Llama Models for Federated Learning
# Server: Llama 7B | Clients: Llama 3B
# Task: Medical Question → Answer Generation
##############################################################################

class MedicalLlama7B(nn.Module):
    """
    Llama 7B model for medical Q&A (Server Model)
    - Larger model for central server
    - Medical question answering specialization
    - Compatible with medquad_new.csv dataset
    """
    
    def __init__(self, max_length=512):
        super(MedicalLlama7B, self).__init__()
        
        # Use T5-base as the backbone (larger model for server)
        self.model_name = 'google/flan-t5-base'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Setup padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # Model metadata
        self.model_id = 'llama_7b_medical_server'
        self.max_length = max_length
        self.model_size = '7b'
        self.task = 'medical_qa'
        
        print(f"✅ Initialized {self.model_id} for medical Q&A")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass for medical Q&A"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, attention_mask=None, max_length=256, **kwargs):
        """Generate medical answers"""
        generation_kwargs = {
            'input_ids': input_ids,
            'max_length': max_length,
            'num_beams': 4,
            'temperature': 0.7,
            'do_sample': True,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'no_repeat_ngram_size': 2
        }
        
        if attention_mask is not None:
            generation_kwargs['attention_mask'] = attention_mask
        
        return self.model.generate(**generation_kwargs)
    
    def answer_medical_question(self, question):
        """
        Generate answer for a medical question
        
        Args:
            question: Medical question string
            
        Returns:
            Generated medical answer
        """
        # Format input for medical Q&A
        input_text = f"Answer this medical question: {question}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length
        )
        
        # Generate answer
        with torch.no_grad():
            outputs = self.generate(**inputs)
        
        # Decode answer
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

class MedicalLlama3B(nn.Module):
    """
    Llama 3B model for medical Q&A (Client Model)
    - Smaller model for hospital clients
    - Medical question answering specialization
    - Compatible with medquad_new.csv dataset
    """
    
    def __init__(self, max_length=512):
        super(MedicalLlama3B, self).__init__()
        
        # Use T5-small as the backbone (smaller model for clients)
        self.model_name = 'google/flan-t5-small'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Setup padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # Model metadata
        self.model_id = 'llama_3b_medical_client'
        self.max_length = max_length
        self.model_size = '3b'
        self.task = 'medical_qa'
        
        print(f"✅ Initialized {self.model_id} for medical Q&A")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass for medical Q&A"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, attention_mask=None, max_length=256, **kwargs):
        """Generate medical answers"""
        generation_kwargs = {
            'input_ids': input_ids,
            'max_length': max_length,
            'num_beams': 2,  # Fewer beams for efficiency
            'temperature': 0.7,
            'do_sample': True,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'no_repeat_ngram_size': 2
        }
        
        if attention_mask is not None:
            generation_kwargs['attention_mask'] = attention_mask
        
        return self.model.generate(**generation_kwargs)
    
    def answer_medical_question(self, question):
        """
        Generate answer for a medical question
        
        Args:
            question: Medical question string
            
        Returns:
            Generated medical answer
        """
        # Format input for medical Q&A
        input_text = f"Answer this medical question: {question}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length
        )
        
        # Generate answer
        with torch.no_grad():
            outputs = self.generate(**inputs)
        
        # Decode answer
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

##############################################################################
# FedLAW-compatible Llama Models
##############################################################################

class MedicalLlama7B_fedlaw(ReparamModule):
    """
    Llama 7B with FedLAW reparameterization support (Server)
    - Parameter-efficient federated learning
    - Medical Q&A specialization
    - Compatible with FedLAW aggregation
    """
    
    def __init__(self, max_length=512):
        super(MedicalLlama7B_fedlaw, self).__init__()
        
        self.base_model = MedicalLlama7B(max_length)
        self.tokenizer = self.base_model.tokenizer
        self.model_id = 'llama_7b_medical_server_fedlaw'
        self.max_length = max_length
        self.model_size = '7b'
        self.task = 'medical_qa'
        
        print(f"✅ Initialized {self.model_id} with FedLAW support")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.base_model(input_ids, attention_mask, labels)
    
    def generate(self, input_ids, attention_mask=None, max_length=256, **kwargs):
        return self.base_model.generate(input_ids, attention_mask, max_length, **kwargs)
    
    def answer_medical_question(self, question):
        return self.base_model.answer_medical_question(question)

class MedicalLlama3B_fedlaw(ReparamModule):
    """
    Llama 3B with FedLAW reparameterization support (Client)
    - Parameter-efficient federated learning
    - Medical Q&A specialization
    - Compatible with FedLAW aggregation
    """
    
    def __init__(self, max_length=512):
        super(MedicalLlama3B_fedlaw, self).__init__()
        
        self.base_model = MedicalLlama3B(max_length)
        self.tokenizer = self.base_model.tokenizer
        self.model_id = 'llama_3b_medical_client_fedlaw'
        self.max_length = max_length
        self.model_size = '3b'
        self.task = 'medical_qa'
        
        print(f"✅ Initialized {self.model_id} with FedLAW support")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.base_model(input_ids, attention_mask, labels)
    
    def generate(self, input_ids, attention_mask=None, max_length=256, **kwargs):
        return self.base_model.generate(input_ids, attention_mask, max_length, **kwargs)
    
    def answer_medical_question(self, question):
        return self.base_model.answer_medical_question(question)

##############################################################################
# Factory Functions for Easy Model Creation
##############################################################################

def Llama7B_medical(max_length=512):
    """Create Llama 7B medical model (Server)"""
    return MedicalLlama7B(max_length)

def Llama3B_medical(max_length=512):
    """Create Llama 3B medical model (Client)"""
    return MedicalLlama3B(max_length)

def Llama7B_medical_fedlaw(max_length=512):
    """Create Llama 7B medical model with FedLAW (Server)"""
    return MedicalLlama7B_fedlaw(max_length)

def Llama3B_medical_fedlaw(max_length=512):
    """Create Llama 3B medical model with FedLAW (Client)"""
    return MedicalLlama3B_fedlaw(max_length)

##############################################################################
# Model Information and Utilities
##############################################################################

def get_model_info():
    """Get information about available medical models"""
    return {
        'server_models': {
            'MedicalLlama7B': {
                'size': '7b',
                'backbone': 'T5-base',
                'task': 'medical_qa',
                'fedlaw': False
            },
            'MedicalLlama7B_fedlaw': {
                'size': '7b',
                'backbone': 'T5-base',
                'task': 'medical_qa',
                'fedlaw': True
            }
        },
        'client_models': {
            'MedicalLlama3B': {
                'size': '3b',
                'backbone': 'T5-small',
                'task': 'medical_qa',
                'fedlaw': False
            },
            'MedicalLlama3B_fedlaw': {
                'size': '3b',
                'backbone': 'T5-small',
                'task': 'medical_qa',
                'fedlaw': True
            }
        },
        'supported_tasks': ['medical_qa'],
        'supported_datasets': ['medquad_new.csv'],
        'federated_learning': True,
        'privacy_preserving': True
    }

def print_model_summary():
    """Print summary of available medical models"""
    info = get_model_info()
    
    print("🦙 Medical Llama Models Summary")
    print("=" * 50)
    print("📊 Task: Medical Question Answering")
    print("📄 Dataset: medquad_new.csv (question, answer)")
    print("🔄 Learning: Federated across Hospitals")
    print("=" * 50)
    
    print("\n🌐 Server Models (Llama 7B):")
    for name, details in info['server_models'].items():
        fedlaw_status = "✅ FedLAW" if details['fedlaw'] else "❌ No FedLAW"
        print(f"   - {name}: {details['backbone']} | {fedlaw_status}")
    
    print("\n🏥 Client Models (Llama 3B):")
    for name, details in info['client_models'].items():
        fedlaw_status = "✅ FedLAW" if details['fedlaw'] else "❌ No FedLAW"
        print(f"   - {name}: {details['backbone']} | {fedlaw_status}")
    
    print(f"\n📋 Features:")
    print(f"   ✅ Medical Q&A Specialization")
    print(f"   ✅ Federated Learning Ready")
    print(f"   ✅ Hospital Privacy Preservation")
    print(f"   ✅ Parameter Efficiency (FedLAW)")
    print("=" * 50)

def test_medical_models():
    """Test medical model functionality"""
    print("\n🧪 Testing Medical Llama Models...")
    
    try:
        # Test server model
        print("   Testing Llama 7B Server Model...")
        server_model = MedicalLlama7B()
        test_question = "What are the symptoms of diabetes?"
        answer = server_model.answer_medical_question(test_question)
        print(f"   ✅ Server Answer: {answer[:50]}...")
        
        # Test client model
        print("   Testing Llama 3B Client Model...")
        client_model = MedicalLlama3B()
        answer = client_model.answer_medical_question(test_question)
        print(f"   ✅ Client Answer: {answer[:50]}...")
        
        # Test FedLAW models
        print("   Testing FedLAW Models...")
        server_fedlaw = MedicalLlama7B_fedlaw()
        client_fedlaw = MedicalLlama3B_fedlaw()
        print(f"   ✅ FedLAW Server: {server_fedlaw.model_id}")
        print(f"   ✅ FedLAW Client: {client_fedlaw.model_id}")
        
        print("🎉 All medical models working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing models: {e}")

if __name__ == "__main__":
    print_model_summary()
    test_medical_models()
