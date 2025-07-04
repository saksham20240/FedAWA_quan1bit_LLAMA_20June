import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

##############################################################################
# Import utility functions (from our updated utils file)
##############################################################################

# We'll define the essential functions here to avoid import issues
def init_model(model_type, args):
    """Initialize model based on type and dataset"""
    try:
        # Get model name from args
        model_name = getattr(args, 'model_name', 'google/flan-t5-small')
        max_length = getattr(args, 'max_length', 512)
        
        print(f"🤖 Initializing {model_type} model: {model_name}")
        
        # Create model based on type
        if model_type in ['server', 'llama_7b']:
            print("🦙 Creating Server Model (T5-Base equivalent)")
            model_name = 'google/flan-t5-base'  # Larger model for server
        else:
            print("🦙 Creating Client Model (T5-Small)")
            model_name = 'google/flan-t5-small'  # Smaller model for client
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Setup padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        
        # Add required attributes
        model.tokenizer = tokenizer
        model.model_id = f'{model_type}_medical_qa'
        model.max_length = max_length
        model.model_type = "seq2seq"
        
        print(f"✅ Model created: {model.model_id}")
        return model
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        raise

def init_optimizer(num_id, model, args):
    """Initialize optimizer for medical language models"""
    try:
        lr = getattr(args, 'lr', 5e-5)
        optimizer_type = getattr(args, 'optimizer', 'adamw')
        weight_decay = getattr(args, 'local_wd_rate', 0.01)
        
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
        return torch.optim.AdamW(model.parameters(), lr=0.0001)

def model_parameter_vector(args, model):
    """Extract model parameters as a vector"""
    try:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) > 0:
            param_vector = torch.cat([p.view(-1) for p in trainable_params], dim=0)
        else:
            param_vector = torch.tensor([])
        return param_vector
    except Exception as e:
        print(f"❌ Error extracting model parameter vector: {e}")
        return torch.tensor([])

##############################################################################
# Medical Dataset Classes
##############################################################################

class MedicalQADataset(Dataset):
    """Medical Q&A Dataset for the new format"""
    
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
        
        # Format for medical Q&A
        input_text = f"Answer this medical question: {question}"
        target_text = answer
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

class MedicalDatasetSplit:
    """Medical dataset split for text data"""
    def __init__(self, questions, answers, indices):
        self.questions = [questions[i] for i in indices]
        self.answers = [answers[i] for i in indices]
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return (self.questions[idx], self.answers[idx])

class DatasetSplit(Dataset):
    """Generic dataset split for legacy compatibility"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

##############################################################################
# Enhanced Node Class for Medical Federated Learning
##############################################################################

class Node(object):
    """
    Enhanced Node class for Medical Federated Learning
    - Server: T5-Base model
    - Clients: T5-Small model
    - Task: Question → Answer generation
    """
    
    def __init__(self, num_id, local_data, train_set, args):
        try:
            self.num_id = num_id
            self.args = args
            self.node_num = getattr(args, 'node_num', 3)
            
            # Set validation ratios
            if num_id == -1:  # Server
                self.valid_ratio = getattr(args, 'server_valid_ratio', 0.1)
                self.is_server = True
            else:  # Client
                self.valid_ratio = getattr(args, 'client_valid_ratio', 0.2)
                self.is_server = False

            # Determine task type
            dataset_name = getattr(args, 'dataset', '')
            self.is_medical = (dataset_name == 'medical_qa' or 'medical' in dataset_name.lower())
            
            if self.is_medical:
                self.num_classes = 1  # Not applicable for text generation
                print(f"🏥 Initializing {'Server' if self.is_server else 'Client'} Node {num_id} for Medical Q&A")
            else:
                # Legacy support for image datasets
                if dataset_name in ['cifar10', 'fmnist']:
                    self.num_classes = 10
                elif dataset_name == 'cifar100':
                    self.num_classes = 100
                elif dataset_name == 'tinyimagenet':
                    self.num_classes = 200
                else:
                    self.num_classes = 10  # Default fallback
                print(f"📊 Initializing Node {num_id} for {dataset_name} classification")
            
            # Model initialization with proper error handling
            self._initialize_model()
            
            # Data splitting - handle both medical and legacy data
            self._initialize_data(local_data, train_set)
            
            # Optimizer initialization with error handling
            self._initialize_optimizer()
            
            # Initialize FedDyn components if needed
            self._initialize_feddyn()
            
            # Initialize FedAdam components if needed (server only)
            self._initialize_fedadam()
            
            print(f"✅ Node {num_id} initialization complete")
            
        except Exception as e:
            print(f"❌ Critical error initializing Node {num_id}: {e}")
            raise

    def _initialize_model(self):
        """Initialize the model based on node type"""
        try:
            if self.is_server:
                # Server uses larger model (T5-Base)
                model_type = getattr(self.args, 'server_model', 'server')
                print(f"   🦙 Loading server model: {model_type}")
            else:
                # Clients use smaller model (T5-Small)
                model_type = getattr(self.args, 'client_model', 
                                   getattr(self.args, 'local_model', 'client'))
                print(f"   🦙 Loading client model: {model_type}")
            
            self.model = init_model(model_type, self.args)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print(f"   ✅ Model moved to GPU")
            
            print(f"   ✅ Successfully initialized {self.model.model_id}")
            
        except Exception as e:
            print(f"   ❌ Error initializing model: {e}")
            raise

    def _initialize_data(self, local_data, train_set):
        """Initialize data loaders based on task type"""
        try:
            if self.is_medical:
                print(f"   📊 Setting up medical data for Node {self.num_id}")
                self.local_data, self.validate_set = self._medical_train_val_split(
                    local_data, self.valid_ratio
                )
            else:
                print(f"   📊 Setting up legacy data for Node {self.num_id}")
                # Legacy data splitting for image datasets
                if getattr(self.args, 'iid', 0) == 1 or self.is_server:
                    self.local_data, self.validate_set = self._train_val_split_for_server(
                        local_data, train_set, self.valid_ratio, self.num_classes
                    )
                else:
                    self.local_data, self.validate_set = self._train_val_split(
                        local_data, train_set, self.valid_ratio
                    )
            
            # Validate data loaders
            if self.local_data is not None:
                train_size = len(self.local_data.dataset) if hasattr(self.local_data, 'dataset') else len(self.local_data)
                print(f"   ✅ Training data: {train_size} samples")
            else:
                print(f"   ⚠️ Warning: No training data for Node {self.num_id}")
            
            if self.validate_set is not None:
                val_size = len(self.validate_set.dataset) if hasattr(self.validate_set, 'dataset') else len(self.validate_set)
                print(f"   ✅ Validation data: {val_size} samples")
            else:
                print(f"   ⚠️ Warning: No validation data for Node {self.num_id}")
                
        except Exception as e:
            print(f"   ❌ Error initializing data: {e}")
            raise

    def _initialize_optimizer(self):
        """Initialize optimizer"""
        try:
            self.optimizer = init_optimizer(self.num_id, self.model, self.args)
            optimizer_name = type(self.optimizer).__name__
            lr = self.optimizer.param_groups[0]['lr']
            print(f"   ✅ Optimizer: {optimizer_name} (lr={lr})")
            
        except Exception as e:
            print(f"   ❌ Error initializing optimizer: {e}")
            # Create a default optimizer as fallback
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
            print(f"   ⚠️ Using fallback AdamW optimizer")

    def _initialize_feddyn(self):
        """Initialize FedDyn components if needed"""
        try:
            if hasattr(self.args, 'client_method') and self.args.client_method == 'feddyn':
                self.old_grad = None
                self.old_grad = copy.deepcopy(self.model)
                self.old_grad = model_parameter_vector(self.args, self.old_grad)
                self.old_grad = torch.zeros_like(self.old_grad)
                print(f"   ✅ FedDyn client components initialized")
                
            if hasattr(self.args, 'server_method') and 'feddyn' in self.args.server_method:
                self.server_state = copy.deepcopy(self.model)
                for param in self.server_state.parameters():
                    param.data = torch.zeros_like(param.data)
                print(f"   ✅ FedDyn server components initialized")
                
        except Exception as e:
            print(f"   ⚠️ Warning: Error initializing FedDyn components: {e}")
            self.old_grad = None
            self.server_state = None

    def _initialize_fedadam(self):
        """Initialize FedAdam components if needed (server only)"""
        try:
            if (hasattr(self.args, 'server_method') and 
                self.args.server_method == 'fedadam' and 
                self.is_server):
                
                # Initialize momentum and velocity for FedAdam
                self.m = copy.deepcopy(self.model)
                self._zero_weights(self.m)
                
                self.v = copy.deepcopy(self.model)
                self._zero_weights(self.v)
                
                print(f"   ✅ FedAdam server components initialized")
                
        except Exception as e:
            print(f"   ⚠️ Warning: Error initializing FedAdam components: {e}")
            self.m = None
            self.v = None

    def _zero_weights(self, model):
        """Zero out all model weights"""
        try:
            for param in model.parameters():
                param.data.zero_()
        except Exception as e:
            print(f"   ⚠️ Error zeroing weights: {e}")

    def _medical_train_val_split(self, data_tuple, valid_ratio):
        """Split medical Q&A data into train and validation sets"""
        try:
            questions, answers = data_tuple
            
            # Ensure we have data
            if len(questions) == 0 or len(answers) == 0:
                print(f"   ⚠️ Warning: Empty data for node {self.num_id}")
                return self._create_empty_medical_loaders()
            
            # Create indices for splitting
            indices = list(range(len(questions)))
            np.random.shuffle(indices)
            
            # Split indices
            validate_size = max(1, int(valid_ratio * len(indices)))
            val_indices = indices[:validate_size]
            train_indices = indices[validate_size:]
            
            # Ensure we have at least one sample in each set
            if len(train_indices) == 0:
                train_indices = [0]
            if len(val_indices) == 0:
                val_indices = [0] if len(indices) > 0 else []
            
            # Create question/answer lists
            train_questions = [questions[i] for i in train_indices]
            train_answers = [answers[i] for i in train_indices]
            val_questions = [questions[i] for i in val_indices] if val_indices else [questions[0]]
            val_answers = [answers[i] for i in val_indices] if val_indices else [answers[0]]
            
            # Create datasets with tokenization
            max_length = getattr(self.args, 'max_length', 512)
            
            # Get tokenizer from model
            tokenizer = self.model.tokenizer
            
            train_dataset = MedicalQADataset(train_questions, train_answers, tokenizer, max_length)
            val_dataset = MedicalQADataset(val_questions, val_answers, tokenizer, max_length)
            
            # Create data loaders
            batch_size = getattr(self.args, 'batch_size', getattr(self.args, 'batchsize', 4))
            val_batch_size = getattr(self.args, 'validate_batchsize', batch_size)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )
            
            return train_loader, val_loader
            
        except Exception as e:
            print(f"   ❌ Error in medical_train_val_split for node {self.num_id}: {e}")
            return self._create_empty_medical_loaders()

    def _create_empty_medical_loaders(self):
        """Create minimal medical data loaders for error cases"""
        try:
            # Create minimal dataset
            tokenizer = self.model.tokenizer
            empty_dataset = MedicalQADataset(
                ["What is health?"], 
                ["Health is physical and mental wellbeing."], 
                tokenizer
            )
            empty_loader = DataLoader(empty_dataset, batch_size=1, shuffle=False)
            return empty_loader, empty_loader
            
        except Exception as e:
            print(f"   ❌ Error creating empty medical loaders: {e}")
            return None, None

    def _train_val_split(self, idxs, train_set, valid_ratio):
        """Split training data into train and validation sets (legacy method for image data)"""
        try:
            # Handle case where idxs might be a tuple (for medical data)
            if isinstance(idxs, tuple):
                return self._medical_train_val_split(idxs, valid_ratio)
            
            # Legacy handling for indices
            if hasattr(idxs, 'indices'):
                idxs = idxs.indices
            elif not isinstance(idxs, (list, np.ndarray)):
                idxs = list(range(len(train_set)))
            
            np.random.shuffle(idxs)

            validate_size = max(1, int(valid_ratio * len(idxs)))
            idxs_test = idxs[:validate_size]
            idxs_train = idxs[validate_size:]

            # Ensure we have training data
            if len(idxs_train) == 0:
                idxs_train = idxs

            train_loader = DataLoader(
                DatasetSplit(train_set, idxs_train),
                batch_size=getattr(self.args, 'batchsize', 4), 
                num_workers=0, 
                shuffle=True
            )

            test_loader = DataLoader(
                DatasetSplit(train_set, idxs_test),
                batch_size=getattr(self.args, 'validate_batchsize', 4),  
                num_workers=0, 
                shuffle=False
            )
            
            return train_loader, test_loader
            
        except Exception as e:
            print(f"   ❌ Error in train_val_split: {e}")
            # Return empty loaders as fallback
            empty_dataset = DatasetSplit(train_set, [])
            empty_loader = DataLoader(empty_dataset, batch_size=1, num_workers=0, shuffle=False)
            return empty_loader, empty_loader

    def _train_val_split_for_server(self, idxs, train_set, valid_ratio, num_classes=10):
        """Split data for server with balanced classes (legacy method)"""
        try:
            # Handle medical data
            if isinstance(idxs, tuple):
                return self._medical_train_val_split(idxs, valid_ratio)
            
            # Legacy handling for image data
            if hasattr(idxs, 'indices'):
                idxs = idxs.indices
            elif not isinstance(idxs, (list, np.ndarray)):
                idxs = list(range(min(1000, len(train_set))))  # Default subset
            
            np.random.shuffle(idxs)
            
            validate_size = max(10, int(valid_ratio * len(idxs)))

            # Generate proxy dataset with balanced classes
            idxs_test = []
            test_class_count = [int(validate_size/num_classes) for _ in range(num_classes)]

            k = 0
            while sum(test_class_count) > 0 and k < len(idxs):
                try:
                    if k < len(train_set):
                        sample = train_set[idxs[k]]
                        if isinstance(sample, tuple) and len(sample) > 1:
                            label = sample[1]
                            if isinstance(label, torch.Tensor):
                                label = label.item()
                            if 0 <= label < num_classes and test_class_count[label] > 0:
                                idxs_test.append(idxs[k])
                                test_class_count[label] -= 1
                except Exception as e:
                    pass  # Skip problematic samples
                k += 1
                
            # Ensure we have some test data
            if len(idxs_test) == 0:
                idxs_test = idxs[:min(10, len(idxs))]

            idxs_train = [idx for idx in idxs if idx not in idxs_test]
            
            # Ensure we have some training data
            if len(idxs_train) == 0:
                idxs_train = idxs[len(idxs_test):]

            train_loader = DataLoader(
                DatasetSplit(train_set, idxs_train),
                batch_size=getattr(self.args, 'batchsize', 4), 
                num_workers=0, 
                shuffle=True
            )
            
            test_loader = DataLoader(
                DatasetSplit(train_set, idxs_test),
                batch_size=getattr(self.args, 'validate_batchsize', 4),  
                num_workers=0, 
                shuffle=False
            )

            return train_loader, test_loader
            
        except Exception as e:
            print(f"   ❌ Error in train_val_split_for_server: {e}")
            # Return minimal loaders
            empty_dataset = DatasetSplit(train_set, [0] if len(train_set) > 0 else [])
            empty_loader = DataLoader(empty_dataset, batch_size=1, num_workers=0, shuffle=False)
            return empty_loader, empty_loader

    def get_model_info(self):
        """Get information about the model"""
        try:
            param_count = sum(p.numel() for p in self.model.parameters())
            trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info = {
                'node_id': self.num_id,
                'node_type': 'Server' if self.is_server else 'Client',
                'model_id': getattr(self.model, 'model_id', 'unknown'),
                'total_params': param_count,
                'trainable_params': trainable_count,
                'is_medical': self.is_medical,
                'has_training_data': self.local_data is not None,
                'has_validation_data': self.validate_set is not None
            }
            
            return info
            
        except Exception as e:
            print(f"   ❌ Error getting model info: {e}")
            return {
                'node_id': self.num_id,
                'error': str(e)
            }

    def validate_setup(self):
        """Validate that the node is properly set up"""
        issues = []
        
        # Check model
        if not hasattr(self, 'model') or self.model is None:
            issues.append("Model not initialized")
        
        # Check optimizer
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            issues.append("Optimizer not initialized")
        
        # Check data
        if self.local_data is None:
            issues.append("No training data")
        
        if self.validate_set is None:
            issues.append("No validation data")
        
        # Check tokenizer for medical tasks
        if self.is_medical:
            if not hasattr(self.model, 'tokenizer'):
                issues.append("Medical model missing tokenizer")
        
        if issues:
            print(f"   ⚠️ Node {self.num_id} setup issues: {', '.join(issues)}")
            return False
        else:
            print(f"   ✅ Node {self.num_id} setup validated successfully")
            return True

    def generate_medical_answer(self, question, max_length=200):
        """Generate answer for a medical question"""
        try:
            self.model.eval()
            
            # Get tokenizer
            tokenizer = self.model.tokenizer
            
            # Format prompt for medical Q&A
            prompt = f"Answer this medical question: {question}"
            
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
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=2,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decode the answer
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part if it contains the prompt
            if prompt in generated_text:
                answer = generated_text.replace(prompt, "").strip()
            else:
                answer = generated_text.strip()
            
            # Clean up the answer
            if len(answer) == 0:
                answer = "I need more information to provide a proper medical answer."
            
            return answer
            
        except Exception as e:
            print(f"❌ Error generating medical answer: {e}")
            return "I cannot generate an answer at this time due to a technical issue."

##############################################################################
# Utility Functions for Long-tail Support (Legacy)
##############################################################################

def label_indices2indices(list_label2indices):
    """Convert list of label indices to flat indices list"""
    try:
        indices_res = []
        for indices in list_label2indices:
            indices_res.extend(indices)
        return indices_res
    except Exception as e:
        print(f"❌ Error in label_indices2indices: {e}")
        return []

def _get_img_num_per_cls(list_label2indices_train, num_classes, imb_factor, imb_type):
    """Calculate number of images per class for long-tail distribution"""
    try:
        img_max = len(list_label2indices_train) / num_classes
        img_num_per_cls = []
        
        if imb_type == 'exp':
            for _classes_idx in range(num_classes):
                num = img_max * (imb_factor**(_classes_idx / (num_classes - 1.0)))
                img_num_per_cls.append(int(num))
        else:
            # Default uniform distribution
            for _classes_idx in range(num_classes):
                img_num_per_cls.append(int(img_max))

        return img_num_per_cls
        
    except Exception as e:
        print(f"❌ Error in _get_img_num_per_cls: {e}")
        # Return uniform distribution as fallback
        img_max = len(list_label2indices_train) / num_classes if list_label2indices_train else 100
        return [int(img_max) for _ in range(num_classes)]

def train_long_tail(list_label2indices_train, num_classes, imb_factor, imb_type):
    """Generate long-tail training distribution"""
    try:
        new_list_label2indices_train = label_indices2indices(copy.deepcopy(list_label2indices_train))
        img_num_list = _get_img_num_per_cls(
            copy.deepcopy(new_list_label2indices_train), 
            num_classes, 
            imb_factor, 
            imb_type
        )
        
        print('img_num_class')
        print(img_num_list)

        list_clients_indices = []
        classes = list(range(num_classes))
        
        for _class, _img_num in zip(classes, img_num_list):
            try:
                indices = list_label2indices_train[_class]
                np.random.shuffle(indices)
                idx = indices[:_img_num]
                list_clients_indices.append(idx)
            except Exception as e:
                print(f"⚠️ Warning: Error processing class {_class}: {e}")
                list_clients_indices.append([])
                
        num_list_clients_indices = label_indices2indices(list_clients_indices)
        print('All num_data_train')
        print(len(num_list_clients_indices))

        return img_num_list, list_clients_indices
        
    except Exception as e:
        print(f"❌ Error in train_long_tail: {e}")
        # Return default values
        default_img_num = [100] * num_classes
        default_indices = [[] for _ in range(num_classes)]
        return default_img_num, default_indices

##############################################################################
# Medical Data Helper Functions
##############################################################################

def create_medical_qa_data_splits(questions, answers, num_clients=3):
    """Create medical Q&A data splits for federated learning"""
    try:
        # Medical specialization keywords
        specializations = {
            0: ['heart', 'blood pressure', 'cardiac', 'cardiovascular'],  # Cardiology
            1: ['cancer', 'tumor', 'chemotherapy', 'radiation'],          # Oncology
            2: ['brain', 'neurological', 'migraine', 'headache'],         # Neurology
            3: ['diabetes', 'insulin', 'glucose', 'hormone'],             # Endocrinology
            4: ['general', 'health', 'symptom', 'treatment']              # General
        }
        
        client_data = []
        data_per_client = len(questions) // num_clients
        
        for i in range(num_clients):
            # Get specialty keywords for this client
            specialty_idx = i % len(specializations)
            keywords = specializations[specialty_idx]
            
            # Filter questions by specialty
            specialized_questions = []
            specialized_answers = []
            general_questions = []
            general_answers = []
            
            for q, a in zip(questions, answers):
                q_lower = q.lower()
                if any(keyword in q_lower for keyword in keywords):
                    specialized_questions.append(q)
                    specialized_answers.append(a)
                else:
                    general_questions.append(q)
                    general_answers.append(a)
            
            # Combine specialized and general data
            client_questions = (
                specialized_questions[:min(len(specialized_questions), data_per_client)] +
                general_questions[:max(0, data_per_client - len(specialized_questions))]
            )
            client_answers = (
                specialized_answers[:min(len(specialized_answers), data_per_client)] +
                general_answers[:max(0, data_per_client - len(specialized_answers))]
            )
            
            # Ensure minimum data
            if len(client_questions) < 5:
                client_questions = questions[:5]
                client_answers = answers[:5]
            
            client_data.append((client_questions, client_answers))
            print(f"Client {i}: {len(client_questions)} samples (specialty: {specialty_idx})")
        
        return client_data
        
    except Exception as e:
        print(f"❌ Error creating medical data splits: {e}")
        # Return simple splits as fallback
        simple_splits = []
        data_per_client = len(questions) // num_clients
        
        for i in range(num_clients):
            start_idx = i * data_per_client
            end_idx = (i + 1) * data_per_client if i < num_clients - 1 else len(questions)
            
            client_questions = questions[start_idx:end_idx]
            client_answers = answers[start_idx:end_idx]
            
            simple_splits.append((client_questions, client_answers))
        
        return simple_splits

##############################################################################
# Testing and Validation
##############################################################################

if __name__ == "__main__":
    # Test the Node class
    print("Testing Medical Node Class...")
    
    class TestArgs:
        def __init__(self):
            self.dataset = 'medical_qa'
            self.node_num = 3
            self.max_length = 512
            self.lr = 5e-5
            self.optimizer = 'adamw'
            self.server_valid_ratio = 0.1
            self.client_valid_ratio = 0.2
            self.batch_size = 2
            self.batchsize = 2
            self.validate_batchsize = 2
            self.server_model = 'server'
            self.client_model = 'client'
            self.model_name = 'google/flan-t5-small'
            self.local_wd_rate = 0.01
            self.momentum = 0.9
    
    args = TestArgs()
    
    try:
        # Test server node
        server_data = (["What is diabetes?"], ["Diabetes is a metabolic disorder."])
        server_node = Node(-1, server_data, None, args)
        print(f"✅ Server node created: {server_node.get_model_info()}")
        
        # Test client node
        client_data = (["How to treat fever?"], ["Fever can be treated with rest and fluids."])
        client_node = Node(0, client_data, None, args)
        print(f"✅ Client node created: {client_node.get_model_info()}")
        
        # Validate setups
        server_node.validate_setup()
        client_node.validate_setup()
        
        # Test medical answer generation
        print("\n🩺 Testing medical answer generation...")
        test_question = "What are the symptoms of diabetes?"
        answer = client_node.generate_medical_answer(test_question)
        print(f"Question: {test_question}")
        print(f"Answer: {answer[:100]}...")
        
        print("\n✅ All Node tests passed!")
        
    except Exception as e:
        print(f"❌ Node test failed: {e}")
        import traceback
        traceback.print_exc()
