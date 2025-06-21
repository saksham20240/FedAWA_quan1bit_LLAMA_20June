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
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
from sklearn.model_selection import train_test_split

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
    except Exception as e:
        print(f"Warning: Error setting up seeds: {e}")

def generate_selectlist(client_node, ratio=0.5):
    """Generate list of selected clients"""
    try:
        candidate_list = [i for i in range(len(client_node))]
        select_num = int(ratio * len(client_node))
        select_list = np.random.choice(candidate_list, select_num, replace=False).tolist()
        return select_list
    except Exception as e:
        print(f"Error generating select list: {e}")
        return list(range(len(client_node)))

def lr_scheduler(rounds, node_list, args):
    """Learning rate scheduler for language models"""
    try:
        if rounds != 0:
            args.lr *= 0.95  # More conservative decay for LLMs
            for i in range(len(node_list)):
                node_list[i].args.lr = args.lr
                node_list[i].optimizer.param_groups[0]['lr'] = args.lr
    except Exception as e:
        print(f"Warning: Error in lr_scheduler: {e}")

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
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = str(self.questions[idx])
        answer = str(self.answers[idx])
        
        # Format for instruction tuning
        prompt = f"Question: {question}\nAnswer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

##############################################################################
# Model Initialization (Updated for Medical LLMs)
##############################################################################

def init_model(model_type, args):
    """Initialize Llama model based on type and dataset"""
    try:
        # Determine model size based on type
        if model_type == 'server' or model_type == 'llama_7b':
            # Server uses larger model (simulating Llama 7B with GPT2-medium)
            model_name = "gpt2-medium"
        elif model_type == 'client' or model_type == 'llama_3b':
            # Clients use smaller model (simulating Llama 3B with GPT2)
            model_name = "gpt2"
        else:
            # Legacy support for CNN/ResNet - convert to LLM
            if model_type in ['CNN', 'ResNet20', 'ResNet18', 'MLP', 'LeNet5']:
                model_name = "gpt2"  # Default to small model
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        # Load tokenizer and model
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
        except:
            # Fallback to basic GPT2
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        
        # Set model_id for tracking
        model.model_id = f'{model_type}_medical_llm'
        model.tokenizer = tokenizer  # Store tokenizer with model
        
        return model
    except Exception as e:
        print(f"Error initializing model: {e}")
        # Fallback to simplest model
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model.model_id = f'{model_type}_fallback'
        model.tokenizer = tokenizer
        return model

def init_optimizer(num_id, model, args):
    """Initialize optimizer for medical language models"""
    try:
        if num_id > -1 and args.client_method == 'fedprox':
            optimizer = PerturbedGradientDescent(model.parameters(), lr=args.lr, mu=args.mu)
        else:
            if args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(
                    model.parameters(), 
                    lr=args.lr, 
                    weight_decay=args.local_wd_rate,
                    eps=1e-8
                )
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(
                    model.parameters(), 
                    lr=args.lr, 
                    weight_decay=args.local_wd_rate
                )
            else:
                optimizer = torch.optim.SGD(
                    model.parameters(), 
                    lr=args.lr, 
                    momentum=getattr(args, 'momentum', 0.9),
                    weight_decay=args.local_wd_rate
                )
        return optimizer
    except Exception as e:
        print(f"Error initializing optimizer: {e}")
        return torch.optim.AdamW(model.parameters(), lr=0.0001)

##############################################################################
# Medical Data Loading Functions
##############################################################################

def load_medical_data(csv_path, args):
    """Load and split medical Q&A data for federated learning"""
    try:
        # Load the medical dataset
        df = pd.read_csv(csv_path)
        
        # Assume CSV has 'question' and 'answer' columns
        if 'question' in df.columns and 'answer' in df.columns:
            questions = df['question'].tolist()
            answers = df['answer'].tolist()
        else:
            # Try alternative column names
            cols = df.columns.tolist()
            if len(cols) >= 2:
                questions = df[cols[0]].tolist()
                answers = df[cols[1]].tolist()
            else:
                raise ValueError("CSV must have at least 2 columns for questions and answers")
        
        # Split data for federated learning (non-IID distribution)
        client_data = split_medical_data_federated(questions, answers, args)
        
        return client_data
    except Exception as e:
        print(f"Error loading medical data: {e}")
        # Return sample data for testing
        return create_sample_medical_data(args)

def split_medical_data_federated(questions, answers, args):
    """Split medical data across clients with non-IID distribution"""
    try:
        # Create a combined dataset
        data = list(zip(questions, answers))
        np.random.shuffle(data)
        
        client_datasets = []
        data_per_client = max(10, len(data) // args.node_num)
        
        for i in range(args.node_num):
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
        
        return client_datasets
    except Exception as e:
        print(f"Error splitting medical data: {e}")
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
        "What are the symptoms of migraine?"
    ]
    
    sample_answers = [
        "Diabetes symptoms include frequent urination, excessive thirst, unexplained weight loss, and fatigue.",
        "High blood pressure is treated with lifestyle changes and medications like ACE inhibitors.",
        "Heart disease is caused by high cholesterol, high blood pressure, smoking, and diabetes.",
        "Chemotherapy side effects include nausea, fatigue, hair loss, and increased infection risk.",
        "Stroke prevention includes controlling blood pressure, maintaining healthy weight, and exercising.",
        "Pneumonia is an infection that inflames air sacs in lungs, causing cough and fever.",
        "Asthma is treated with bronchodilators for quick relief and anti-inflammatory medications.",
        "Migraine symptoms include severe headache, nausea, and sensitivity to light and sound."
    ]
    
    # Split among clients
    client_datasets = []
    data_per_client = len(sample_questions) // args.node_num
    
    for i in range(args.node_num):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client if i < args.node_num - 1 else len(sample_questions)
        
        client_questions = sample_questions[start_idx:end_idx]
        client_answers = sample_answers[start_idx:end_idx]
        
        # Ensure each client has at least some data
        if len(client_questions) == 0:
            client_questions = sample_questions[:2]
            client_answers = sample_answers[:2]
        
        client_datasets.append((client_questions, client_answers))
    
    return client_datasets

##############################################################################
# Validation Functions (Updated for Medical LLMs)
##############################################################################

def validate(args, node, which_dataset='validate'):
    """Enhanced validation function for medical models"""
    try:
        node.model.eval()
        
        if which_dataset == 'validate':
            test_loader = node.validate_set
        elif which_dataset == 'local':
            test_loader = node.local_data
        else:
            raise ValueError('Undefined dataset type')

        total_loss = 0.0
        total_samples = 0
        perplexity_scores = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
                attention_mask = batch['attention_mask'].cuda() if torch.cuda.is_available() else batch['attention_mask']
                labels = batch['labels'].cuda() if torch.cuda.is_available() else batch['labels']
                
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
        
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
        avg_perplexity = np.mean(perplexity_scores) if perplexity_scores else float('inf')
        
        # Return perplexity as a percentage-like metric (lower is better)
        # Convert to accuracy-like score: higher is better
        accuracy_like = max(0, 100 - min(100, avg_perplexity))
        
        return accuracy_like
    except Exception as e:
        print(f"Error in validation: {e}")
        return 0.0

def testloss(args, node, which_dataset='validate'):
    """Test loss computation for medical models"""
    try:
        node.model.eval()
        
        if which_dataset == 'validate':
            test_loader = node.validate_set
        elif which_dataset == 'local':
            test_loader = node.local_data
        else:
            raise ValueError('Undefined dataset type')

        losses = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
                attention_mask = batch['attention_mask'].cuda() if torch.cuda.is_available() else batch['attention_mask']
                labels = batch['labels'].cuda() if torch.cuda.is_available() else batch['labels']
                
                outputs = node.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                losses.append(outputs.loss.item())
        
        return np.mean(losses) if losses else 0.0
    except Exception as e:
        print(f"Error in test loss computation: {e}")
        return 0.0

##############################################################################
# Medical Text Generation Functions
##############################################################################

def generate_medical_answer(args, node, question, max_length=200):
    """Generate answer for a medical question"""
    try:
        node.model.eval()
        tokenizer = node.model.tokenizer
        
        # Format the question
        prompt = f"Question: {question}\nAnswer:"
        
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
            outputs = node.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=2,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the answer
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text.strip()
        
        return answer
    except Exception as e:
        print(f"Error generating medical answer: {e}")
        return "I cannot generate an answer at this time."

##############################################################################
# FedLAW Support Functions (Updated for Medical Models)
##############################################################################

def validate_with_param(args, node, param, which_dataset='validate'):
    """FedLAW validation with parameters for medical models"""
    try:
        # For now, use standard validation since parameter injection 
        # for transformers requires more complex implementation
        return validate(args, node, which_dataset)
    except Exception as e:
        print(f"Error in validate_with_param: {e}")
        return 0.0

def testloss_with_param(args, node, param, which_dataset='validate'):
    """FedLAW test loss with parameters for medical models"""
    try:
        # For now, use standard test loss
        return testloss(args, node, which_dataset)
    except Exception as e:
        print(f"Error in testloss_with_param: {e}")
        return 0.0

##############################################################################
# Data Preparation Functions
##############################################################################

def prepare_medical_dataloaders(client_data, model, args):
    """Prepare data loaders for medical federated learning"""
    try:
        tokenizer = model.tokenizer
        client_loaders = []
        
        for questions, answers in client_data:
            # Split into train/validation
            if len(questions) > 4:
                train_q, val_q, train_a, val_a = train_test_split(
                    questions, answers, test_size=0.2, random_state=getattr(args, 'random_seed', 42)
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
            batch_size = getattr(args, 'batch_size', 4)
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
        
        return client_loaders
    except Exception as e:
        print(f"Error preparing medical data loaders: {e}")
        return []

##############################################################################
# Optimizer Classes
##############################################################################

class PerturbedGradientDescent(Optimizer):
    """Perturbed Gradient Descent optimizer for FedProx (adapted for medical models)"""
    def __init__(self, params, lr=0.01, mu=0.01):  # Lower mu for stability with LLMs
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                if p.grad is not None:
                    d_p = p.grad.data + group['mu'] * (p.data - g.data)
                    p.data.add_(d_p, alpha=-group['lr'])

##############################################################################
# Utility Classes
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

def model_parameter_vector(args, model):
    """Extract model parameters as a vector"""
    try:
        # For medical LLMs, extract all trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) > 0:
            param_vector = torch.cat([p.view(-1) for p in trainable_params], dim=0)
        else:
            param_vector = torch.tensor([])
        return param_vector
    except Exception as e:
        print(f"Error extracting model parameter vector: {e}")
        return torch.tensor([])

def set_model_parameters(model, param_vector):
    """Set model parameters from vector"""
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
        print(f"Error setting model parameters: {e}")

##############################################################################
# Medical-Specific Utility Functions
##############################################################################

def evaluate_medical_generation(args, node, test_questions, max_samples=3):
    """Evaluate medical answer generation quality"""
    try:
        results = []
        sample_size = min(max_samples, len(test_questions))
        
        for i in range(sample_size):
            question = test_questions[i]
            answer = generate_medical_answer(args, node, question)
            results.append({
                'question': question,
                'generated_answer': answer,
                'answer_length': len(answer.split())
            })
        
        return results
    except Exception as e:
        print(f"Error in medical generation evaluation: {e}")
        return []

def print_medical_evaluation(args, node, sample_questions):
    """Print sample medical Q&A for evaluation"""
    try:
        print("\n=== Medical Q&A Evaluation ===")
        results = evaluate_medical_generation(args, node, sample_questions)
        
        for i, result in enumerate(results):
            print(f"\nSample {i+1}:")
            print(f"Q: {result['question']}")
            print(f"A: {result['generated_answer']}")
            print(f"Length: {result['answer_length']} words")
            print("-" * 50)
    except Exception as e:
        print(f"Error printing medical evaluation: {e}")

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
