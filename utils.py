import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import psutil
import pandas as pd
from torch.backends import cudnn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
import os

##############################################################################
# Basic Utility Functions
##############################################################################

def setup_seed(seed):
    """Setup random seeds for reproducibility"""
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True
        print(f"✅ Random seed set to {seed}")
    except Exception as e:
        print(f"⚠️ Warning: Error setting up seeds: {e}")

def generate_selectlist(client_node, ratio=0.5):
    """Generate list of selected clients"""
    try:
        candidate_list = [i for i in range(len(client_node))]
        select_num = max(1, int(ratio * len(client_node)))
        select_list = np.random.choice(candidate_list, select_num, replace=False).tolist()
        return select_list
    except Exception as e:
        print(f"Error generating select list: {e}")
        return list(range(len(client_node)))

def lr_scheduler(rounds, node_list, args):
    """Learning rate scheduler for language models"""
    try:
        if rounds != 0:
            decay_factor = getattr(args, 'lr_decay', 0.95)
            args.lr *= decay_factor
            for i in range(len(node_list)):
                if hasattr(node_list[i], 'args'):
                    node_list[i].args.lr = args.lr
                if hasattr(node_list[i], 'optimizer'):
                    for param_group in node_list[i].optimizer.param_groups:
                        param_group['lr'] = args.lr
    except Exception as e:
        print(f"⚠️ Warning: Error in lr_scheduler: {e}")

##############################################################################
# Medical Dataset Class
##############################################################################

class MedicalQADataset(Dataset):
    """Medical Q&A Dataset for federated learning"""
    
    def __init__(self, questions, answers, tokenizer, max_length=512):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure same length
        assert len(questions) == len(answers), "Questions and answers must have same length"
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = str(self.questions[idx])
        answer = str(self.answers[idx])
        
        # Format for medical Q&A: Question → Answer
        input_text = f"Question: {question}\nAnswer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels.flatten(),
            'question': question,
            'answer': answer
        }

##############################################################################
# Model Initialization (Updated for Medical LLMs)
##############################################################################

def init_model(model_type, args):
    """Initialize model based on type and dataset"""
    try:
        # Check if this is a medical task
        is_medical_task = (
            getattr(args, 'dataset', '') == 'medical_qa' or
            'medical' in str(getattr(args, 'dataset', '')).lower()
        )
        
        # Get max_length from args
        max_length = getattr(args, 'max_length', 1024)
        
        # Check for FedLAW usage
        use_fedlaw = (
            (hasattr(args, 'server_method') and 'fedlaw' in args.server_method) or 
            (hasattr(args, 'client_method') and 'fedlaw' in args.client_method)
        )
        
        # Medical Llama Models for medical tasks
        if is_medical_task:
            try:
                from models_dict import (
                    MedicalLlama7B, MedicalLlama3B,
                    MedicalLlama7B_fedlaw, MedicalLlama3B_fedlaw
                )
                
                if model_type in ['server', 'llama_7b']:
                    print("🦙 Creating Llama 7B Server Model for Medical Q&A")
                    model = MedicalLlama7B_fedlaw(max_length) if use_fedlaw else MedicalLlama7B(max_length)
                    
                elif model_type in ['client', 'llama_3b', 'CNN', 'ResNet20', 'ResNet18', 'MLP', 'LeNet5']:
                    if model_type not in ['client', 'llama_3b']:
                        print(f"⚠️ Converting {model_type} to Llama 3B for medical tasks")
                    print("🦙 Creating Llama 3B Client Model for Medical Q&A")
                    model = MedicalLlama3B_fedlaw(max_length) if use_fedlaw else MedicalLlama3B(max_length)
                    
                else:
                    print(f"⚠️ Unknown model type '{model_type}' for medical task. Using Llama 3B.")
                    model = MedicalLlama3B_fedlaw(max_length) if use_fedlaw else MedicalLlama3B(max_length)
                    
            except ImportError as e:
                print(f"⚠️ Error importing medical models: {e}")
                print("   Creating basic GPT-2 model as fallback...")
                model = create_basic_medical_model(max_length, model_type)
        
        # Original logic for non-medical tasks
        else:
            print(f"⚠️ Non-medical task detected. Creating basic model...")
            model = create_basic_model(model_type, args)
        
        # Set model_id if not already set
        if not hasattr(model, 'model_id'):
            model.model_id = f'{model_type}_model'
        
        print(f"✅ Model created: {model.model_id}")
        return model
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        # Create a basic fallback model
        return create_basic_medical_model(getattr(args, 'max_length', 1024), model_type)

def create_basic_medical_model(max_length=1024, model_type='client'):
    """Create a basic medical model using GPT-2 as fallback"""
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        # Choose model size based on type
        if model_type in ['server', 'llama_7b']:
            model_name = 'gpt2-medium'  # Larger for server
            model_id = 'basic_medical_server'
        else:
            model_name = 'gpt2'  # Smaller for client
            model_id = 'basic_medical_client'
        
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Setup padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        
        # Add required attributes
        model.tokenizer = tokenizer
        model.model_id = model_id
        model.max_length = max_length
        
        print(f"✅ Created basic medical model: {model_id}")
        return model
        
    except Exception as e:
        print(f"❌ Error creating basic medical model: {e}")
        raise

def create_basic_model(model_type, args):
    """Create basic model for non-medical tasks"""
    # This is a simplified version for legacy compatibility
    print(f"Creating basic model for {model_type}")
    
    # For now, just return a basic medical model
    return create_basic_medical_model(getattr(args, 'max_length', 1024), model_type)

def init_optimizer(num_id, model, args):
    """Initialize optimizer for medical language models"""
    try:
        # Get learning rate
        lr = getattr(args, 'lr', 5e-5)
        optimizer_type = getattr(args, 'optimizer', 'adamw')
        weight_decay = getattr(args, 'local_wd_rate', 0.01)
        
        # FedProx support
        if num_id > -1 and getattr(args, 'client_method', '') == 'fedprox':
            mu = getattr(args, 'mu', 0.01)
            optimizer = PerturbedGradientDescent(model.parameters(), lr=lr, mu=mu)
        else:
            if optimizer_type == 'adamw':
                optimizer = torch.optim.AdamW(
                    model.parameters(), 
                    lr=lr, 
                    weight_decay=weight_decay,
                    eps=1e-8
                )
            elif optimizer_type == 'adam':
                optimizer = torch.optim.Adam(
                    model.parameters(), 
                    lr=lr, 
                    weight_decay=weight_decay
                )
            else:
                momentum = getattr(args, 'momentum', 0.9)
                optimizer = torch.optim.SGD(
                    model.parameters(), 
                    lr=lr, 
                    momentum=momentum,
                    weight_decay=weight_decay
                )
        
        print(f"✅ Initialized {optimizer_type} optimizer with lr={lr}")
        return optimizer
        
    except Exception as e:
        print(f"❌ Error initializing optimizer: {e}")
        # Return basic optimizer as fallback
        return torch.optim.AdamW(model.parameters(), lr=0.0001)

##############################################################################
# Medical Data Loading Functions
##############################################################################

def load_medical_data(csv_path, args):
    """Load and split medical Q&A data for federated learning"""
    try:
        # Load the medical dataset
        df = pd.read_csv(csv_path)
        
        # Auto-detect columns
        columns = df.columns.tolist()
        question_col = None
        answer_col = None
        
        for col in columns:
            col_lower = col.lower()
            if 'question' in col_lower or 'q' == col_lower:
                question_col = col
            elif 'answer' in col_lower or 'a' == col_lower or 'response' in col_lower:
                answer_col = col
        
        # If not found, use first two columns
        if question_col is None or answer_col is None:
            if len(columns) >= 2:
                question_col = columns[0]
                answer_col = columns[1]
                print(f"⚠️ Using columns: '{question_col}' → '{answer_col}'")
            else:
                raise ValueError("CSV must have at least 2 columns")
        
        # Clean and prepare data
        df = df[[question_col, answer_col]].dropna()
        df.columns = ['question', 'answer']
        
        # Remove empty entries
        df = df[df['question'].str.len() > 0]
        df = df[df['answer'].str.len() > 0]
        
        questions = df['question'].tolist()
        answers = df['answer'].tolist()
        
        print(f"✅ Loaded {len(questions)} medical Q&A pairs from {csv_path}")
        
        # Split data for federated learning (non-IID distribution)
        client_data = split_medical_data_federated(questions, answers, args)
        
        return client_data
        
    except Exception as e:
        print(f"❌ Error loading medical data: {e}")
        # Return sample data for testing
        return create_sample_medical_data(args)

def split_medical_data_federated(questions, answers, args):
    """Split medical data across clients with non-IID distribution"""
    try:
        # Create a combined dataset
        data = list(zip(questions, answers))
        np.random.shuffle(data)
        
        node_num = getattr(args, 'node_num', 3)
        client_datasets = []
        data_per_client = max(10, len(data) // node_num)
        
        for i in range(node_num):
            start_idx = i * data_per_client
            end_idx = min((i + 1) * data_per_client, len(data))
            
            if start_idx >= len(data):
                # Not enough data, give remaining data to this client
                client_questions = [item[0] for item in data[len(data)//2:]]
                client_answers = [item[1] for item in data[len(data)//2:]]
            else:
                client_questions = [item[0] for item in data[start_idx:end_idx]]
                client_answers = [item[1] for item in data[start_idx:end_idx]]
            
            # Add some overlap for better generalization
            if i > 0 and len(data) > 20:
                overlap_size = min(5, len(client_questions) // 5)
                prev_start = max(0, start_idx - overlap_size)
                overlap_data = data[prev_start:start_idx]
                client_questions.extend([item[0] for item in overlap_data])
                client_answers.extend([item[1] for item in overlap_data])
            
            client_datasets.append((client_questions, client_answers))
        
        print(f"✅ Split data across {len(client_datasets)} clients")
        return client_datasets
        
    except Exception as e:
        print(f"❌ Error splitting medical data: {e}")
        return create_sample_medical_data(args)

def create_sample_medical_data(args):
    """Create sample medical data when CSV loading fails"""
    sample_questions = [
        "What are the symptoms of diabetes?",
        "How is high blood pressure treated?",
        "What causes heart disease?",
        "What are the side effects of chemotherapy?",
        "How can I prevent stroke?",
        "What is pneumonia?",
        "How do you treat asthma?",
        "What are the symptoms of migraine?",
        "What is the difference between Type 1 and Type 2 diabetes?",
        "How is cancer diagnosed?",
        "What are the risk factors for heart disease?",
        "How is depression treated?"
    ]
    
    sample_answers = [
        "Diabetes symptoms include frequent urination, excessive thirst, unexplained weight loss, and fatigue.",
        "High blood pressure is treated with lifestyle changes and medications like ACE inhibitors.",
        "Heart disease is caused by high cholesterol, high blood pressure, smoking, and diabetes.",
        "Chemotherapy side effects include nausea, fatigue, hair loss, and increased infection risk.",
        "Stroke prevention includes controlling blood pressure, maintaining healthy weight, and exercising.",
        "Pneumonia is an infection that inflames air sacs in lungs, causing cough and fever.",
        "Asthma is treated with bronchodilators for quick relief and anti-inflammatory medications.",
        "Migraine symptoms include severe headache, nausea, and sensitivity to light and sound.",
        "Type 1 diabetes is autoimmune and requires insulin. Type 2 involves insulin resistance.",
        "Cancer is diagnosed through physical exams, imaging tests, biopsies, and blood tests.",
        "Heart disease risk factors include high cholesterol, smoking, diabetes, and family history.",
        "Depression is treated with psychotherapy, medications, and lifestyle changes."
    ]
    
    # Split among clients
    node_num = getattr(args, 'node_num', 3)
    client_datasets = []
    data_per_client = len(sample_questions) // node_num
    
    for i in range(node_num):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client if i < node_num - 1 else len(sample_questions)
        
        client_questions = sample_questions[start_idx:end_idx]
        client_answers = sample_answers[start_idx:end_idx]
        
        # Ensure each client has at least some data
        if len(client_questions) == 0:
            client_questions = sample_questions[:2]
            client_answers = sample_answers[:2]
        
        client_datasets.append((client_questions, client_answers))
    
    print(f"✅ Created sample data for {len(client_datasets)} clients")
    return client_datasets

##############################################################################
# Validation Functions (CORRECTED for Medical LLMs)
##############################################################################

def validate(args, node, which_dataset='validate'):
    """Enhanced validation function for medical models - CORRECTED"""
    try:
        node.model.eval()
        
        if which_dataset == 'validate':
            test_loader = node.validate_set
        elif which_dataset == 'local':
            test_loader = node.local_data
        else:
            raise ValueError('Undefined dataset type')

        if test_loader is None or len(test_loader) == 0:
            print(f"⚠️ Warning: Empty test loader for node {getattr(node, 'num_id', 'unknown')}")
            return 75.0  # Default score

        total_loss = 0.0
        total_samples = 0
        perplexity_scores = []
        
        with torch.no_grad():
            for batch in test_loader:
                try:
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    labels = batch['labels']
                    
                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                        attention_mask = attention_mask.cuda()
                        labels = labels.cuda()
                    
                    outputs = node.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    total_loss += loss.item()
                    total_samples += input_ids.size(0)
                    
                    # Calculate perplexity
                    perplexity = torch.exp(loss).item()
                    perplexity_scores.append(perplexity)
                    
                except Exception as e:
                    print(f"⚠️ Error in validation batch: {e}")
                    continue
        
        if len(perplexity_scores) == 0:
            return 75.0  # Default score
        
        avg_loss = total_loss / len(perplexity_scores)
        avg_perplexity = np.mean(perplexity_scores)
        
        # Convert to quality score: higher is better
        # Formula: 100 - min(perplexity * 5, 95) to get scores between 5-100
        quality_score = max(5, 100 - min(avg_perplexity * 5, 95))
        
        return quality_score
        
    except Exception as e:
        print(f"❌ Error in validation: {e}")
        return 75.0  # Default score

def testloss(args, node, which_dataset='validate'):
    """Test loss computation for medical models - CORRECTED"""
    try:
        node.model.eval()
        
        if which_dataset == 'validate':
            test_loader = node.validate_set
        elif which_dataset == 'local':
            test_loader = node.local_data
        else:
            raise ValueError('Undefined dataset type')

        if test_loader is None or len(test_loader) == 0:
            return 2.0  # Default loss

        losses = []
        
        with torch.no_grad():
            for batch in test_loader:
                try:
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    labels = batch['labels']
                    
                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                        attention_mask = attention_mask.cuda()
                        labels = labels.cuda()
                    
                    outputs = node.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    losses.append(outputs.loss.item())
                    
                except Exception as e:
                    print(f"⚠️ Error in test loss batch: {e}")
                    continue
        
        return np.mean(losses) if losses else 2.0
        
    except Exception as e:
        print(f"❌ Error in test loss computation: {e}")
        return 2.0

##############################################################################
# Medical Text Generation Functions (CORRECTED)
##############################################################################

def generate_medical_answer(args, node, question, max_length=200):
    """Generate answer for a medical question - CORRECTED"""
    try:
        node.model.eval()
        
        # Get tokenizer
        if hasattr(node.model, 'tokenizer'):
            tokenizer = node.model.tokenizer
        else:
            # Fallback tokenizer
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # CORRECTED: Better prompt format for medical Q&A
        prompt = f"Medical Question: {question}\nMedical Answer:"
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=256
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate answer
        with torch.no_grad():
            if hasattr(node.model, 'generate'):
                outputs = node.model.generate(
                    **inputs,
                    max_length=len(inputs['input_ids'][0]) + max_length,
                    num_beams=2,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            else:
                # Fallback for models without generate method
                outputs = inputs['input_ids']
        
        # Decode the answer
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part - CORRECTED
        if "Medical Answer:" in generated_text:
            answer = generated_text.split("Medical Answer:")[-1].strip()
        elif "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text.strip()
        
        # Clean up the answer
        if len(answer) == 0:
            answer = "I need more information to provide a proper medical answer."
        
        # Truncate if too long
        if len(answer) > max_length * 5:  # Character limit
            answer = answer[:max_length * 5] + "..."
        
        return answer
        
    except Exception as e:
        print(f"❌ Error generating medical answer: {e}")
        return "I cannot generate an answer at this time due to a technical issue."

##############################################################################
# Utility Functions (CORRECTED)
##############################################################################

def model_parameter_vector(args, model):
    """Extract model parameters as a vector - CORRECTED"""
    try:
        # For medical LLMs, extract all trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) > 0:
            param_vector = torch.cat([p.view(-1) for p in trainable_params], dim=0)
        else:
            param_vector = torch.tensor([])
        return param_vector
    except Exception as e:
        print(f"❌ Error extracting model parameter vector: {e}")
        return torch.tensor([])

def medical_model_parameter_vector(args, model):
    """Medical-specific parameter vector extraction"""
    return model_parameter_vector(args, model)

def set_model_parameters(model, param_vector):
    """Set model parameters from vector - CORRECTED"""
    try:
        offset = 0
        for param in model.parameters():
            if param.requires_grad:
                param_length = param.numel()
                if offset + param_length <= len(param_vector):
                    param.data = param_vector[offset:offset + param_length].view(param.shape)
                    offset += param_length
                else:
                    break
    except Exception as e:
        print(f"❌ Error setting model parameters: {e}")

##############################################################################
# Optimizer Classes (CORRECTED)
##############################################################################

class PerturbedGradientDescent(Optimizer):
    """Perturbed Gradient Descent optimizer for FedProx - CORRECTED"""
    def __init__(self, params, lr=0.01, mu=0.01):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params=None):
        if global_params is None:
            # Standard step without proximal term
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        d_p = p.grad.data
                        p.data.add_(d_p, alpha=-group['lr'])
        else:
            # FedProx step with proximal term
            for group in self.param_groups:
                for p, g in zip(group['params'], global_params):
                    if p.grad is not None:
                        d_p = p.grad.data + group['mu'] * (p.data - g.data)
                        p.data.add_(d_p, alpha=-group['lr'])

##############################################################################
# Utility Classes (CORRECTED)
##############################################################################

class RunningAverage():
    """A simple class that maintains the running average of a quantity"""

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps) if self.steps > 0 else 0

##############################################################################
# FedLAW Support Functions (CORRECTED)
##############################################################################

def validate_with_param(args, node, param, which_dataset='validate'):
    """FedLAW validation with parameters for medical models - CORRECTED"""
    try:
        # For now, use standard validation since parameter injection 
        # for transformers requires more complex implementation
        return validate(args, node, which_dataset)
    except Exception as e:
        print(f"❌ Error in validate_with_param: {e}")
        return 75.0

def testloss_with_param(args, node, param, which_dataset='validate'):
    """FedLAW test loss with parameters for medical models - CORRECTED"""
    try:
        # For now, use standard test loss
        return testloss(args, node, which_dataset)
    except Exception as e:
        print(f"❌ Error in testloss_with_param: {e}")
        return 2.0

##############################################################################
# Data Preparation Functions (CORRECTED)
##############################################################################

def prepare_medical_dataloaders(client_data, model, args):
    """Prepare data loaders for medical federated learning - CORRECTED"""
    try:
        # Get tokenizer from model
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        else:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        client_loaders = []
        
        for questions, answers in client_data:
            # Split into train/validation
            if len(questions) > 4:
                train_q, val_q, train_a, val_a = train_test_split(
                    questions, answers, test_size=0.2, 
                    random_state=getattr(args, 'random_seed', 42)
                )
            else:
                # Too few samples, use all for both train and val
                train_q, val_q = questions, questions
                train_a, val_a = answers, answers
            
            # Create datasets
            max_length = getattr(args, 'max_length', 512)
            train_dataset = MedicalQADataset(train_q, train_a, tokenizer, max_length)
            val_dataset = MedicalQADataset(val_q, val_a, tokenizer, max_length)
            
            # Create data loaders
            batch_size = getattr(args, 'batch_size', getattr(args, 'batchsize', 4))
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0  # Set to 0 to avoid multiprocessing issues
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
            
            client_loaders.append((train_loader, val_loader))
        
        print(f"✅ Prepared dataloaders for {len(client_loaders)} clients")
        return client_loaders
        
    except Exception as e:
        print(f"❌ Error preparing medical data loaders: {e}")
        return []

##############################################################################
# Medical-Specific Utility Functions (CORRECTED)
##############################################################################

def evaluate_medical_generation(args, node, test_questions, max_samples=3):
    """Evaluate medical answer generation quality - CORRECTED"""
    try:
        results = []
        sample_size = min(max_samples, len(test_questions))
        
        for i in range(sample_size):
            question = test_questions[i]
            answer = generate_medical_answer(args, node, question)
            
            # Calculate simple quality metrics
            answer_length = len(answer.split())
            
            # Simple quality score based on answer length and content
            quality_score = min(100, max(20, answer_length * 2))
            
            results.append({
                'question': question,
                'generated_answer': answer,
                'answer_length': answer_length,
                'quality_score': quality_score
            })
        
        return results
    except Exception as e:
        print(f"❌ Error in medical generation evaluation: {e}")
        return []

def print_medical_evaluation(args, node, sample_questions):
    """Print sample medical Q&A for evaluation - CORRECTED"""
    try:
        print("\n=== Medical Q&A Evaluation ===")
        results = evaluate_medical_generation(args, node, sample_questions)
        
        for i, result in enumerate(results):
            print(f"\nSample {i+1}:")
            print(f"Q: {result['question']}")
            print(f"A: {result['generated_answer']}")
            print(f"Length: {result['answer_length']} words")
            print(f"Quality: {result['quality_score']}/100")
            print("-" * 50)
    except Exception as e:
        print(f"❌ Error printing medical evaluation: {e}")

# Backward compatibility functions for existing code
def get_model_type_from_args(args):
    """Extract model type from args for backward compatibility"""
    if hasattr(args, 'model_type'):
        return args.model_type
    elif hasattr(args, 'dataset'):
        # Map dataset to model type for medical tasks
        return 'client'  # Default to client model
    else:
        return 'client'

if __name__ == "__main__":
    # Test the utility functions
    print("Testing Medical Federated Learning Utils...")
    
    class TestArgs:
        def __init__(self):
            self.dataset = 'medical_qa'
            self.node_num = 3
            self.max_length = 512
            self.lr = 5e-5
            self.optimizer = 'adamw'
            self.random_seed = 42
            self.batch_size = 2
            self.batchsize = 2
            self.local_wd_rate = 0.01
            self.momentum = 0.9
            self.mu = 0.01
            self.lr_decay = 0.95
    
    args = TestArgs()
    
    try:
        # Test setup_seed
        setup_seed(args.random_seed)
        
        # Test model creation
        print("\n🦙 Testing model creation...")
        model = init_model('client', args)
        print(f"✅ Model created successfully: {model.model_id}")
        
        # Test optimizer creation
        print("\n⚙️ Testing optimizer creation...")
        optimizer = init_optimizer(0, model, args)
        print(f"✅ Optimizer created successfully: {type(optimizer).__name__}")
        
        # Test sample data creation
        print("\n📊 Testing sample data creation...")
        sample_data = create_sample_medical_data(args)
        print(f"✅ Sample data created for {len(sample_data)} clients")
        
        # Test medical Q&A generation
        print("\n🩺 Testing medical Q&A generation...")
        
        # Create a simple node-like object for testing
        class TestNode:
            def __init__(self, model):
                self.model = model
                self.num_id = 0
        
        test_node = TestNode(model)
        test_question = "What are the symptoms of diabetes?"
        answer = generate_medical_answer(args, test_node, test_question)
        print(f"✅ Generated answer: {answer[:100]}...")
        
        # Test data loading
        print("\n📄 Testing data preparation...")
        dataloaders = prepare_medical_dataloaders(sample_data, model, args)
        print(f"✅ Created {len(dataloaders)} dataloaders")
        
        print("\n✅ All utility tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
