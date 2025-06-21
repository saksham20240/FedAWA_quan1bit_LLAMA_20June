import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import init_model, init_optimizer, model_parameter_vector, MedicalQADataset, prepare_medical_dataloaders


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


class Node(object):
    def __init__(self, num_id, local_data, train_set, args):
        try:
            self.num_id = num_id
            self.args = args
            self.node_num = self.args.node_num
            
            if num_id == -1:
                self.valid_ratio = getattr(args, 'server_valid_ratio', 0.1)
            else:
                self.valid_ratio = getattr(args, 'client_valid_ratio', 0.2)

            # For medical Q&A, we don't have traditional class numbers
            # Instead, we work with question-answer pairs
            if hasattr(args, 'dataset') and args.dataset == 'medical_qa':
                self.num_classes = 1  # Not applicable for text generation
                self.is_medical = True
            else:
                # Legacy support for image datasets
                if getattr(args, 'dataset', '') == 'cifar10' or getattr(args, 'dataset', '') == 'fmnist':
                    self.num_classes = 10
                elif getattr(args, 'dataset', '') == 'cifar100':
                    self.num_classes = 100
                elif getattr(args, 'dataset', '') == 'tinyimagenet':
                    self.num_classes = 200
                else:
                    self.num_classes = 10  # Default fallback
                self.is_medical = False
            
            # Model initialization with error handling
            try:
                if num_id == -1:
                    # Server uses larger model
                    self.model = init_model('server', self.args)
                else:
                    # Clients use smaller model
                    self.model = init_model('client', self.args)
                
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                
                print(f"✅ Successfully initialized {'server' if num_id == -1 else 'client'} model for node {num_id}")
            except Exception as e:
                print(f"❌ Error initializing model for node {num_id}: {e}")
                raise
            
            # Data splitting - handle both medical and legacy data
            if self.is_medical:
                self.local_data, self.validate_set = self.medical_train_val_split(
                    local_data, self.valid_ratio
                )
            else:
                # Legacy data splitting for image datasets
                if getattr(args, 'iid', 0) == 1 or num_id == -1:
                    self.local_data, self.validate_set = self.train_val_split_forServer(
                        local_data.indices if hasattr(local_data, 'indices') else local_data, 
                        train_set, self.valid_ratio, self.num_classes
                    )
                else:
                    self.local_data, self.validate_set = self.train_val_split(
                        local_data, train_set, self.valid_ratio
                    )
            
            # Optimizer initialization with error handling
            try:
                self.optimizer = init_optimizer(self.num_id, self.model, args)
                print(f"✅ Successfully initialized optimizer for node {num_id}")
            except Exception as e:
                print(f"❌ Error initializing optimizer for node {num_id}: {e}")
                # Create a default optimizer as fallback
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
            
            # Node init for feddyn
            if hasattr(args, 'client_method') and args.client_method == 'feddyn':
                try:
                    self.old_grad = None
                    self.old_grad = copy.deepcopy(self.model)
                    self.old_grad = model_parameter_vector(args, self.old_grad)
                    self.old_grad = torch.zeros_like(self.old_grad)
                except Exception as e:
                    print(f"Warning: Error initializing FedDyn components for node {num_id}: {e}")
                    self.old_grad = None
                    
            if hasattr(args, 'server_method') and 'feddyn' in args.server_method:
                try:
                    self.server_state = copy.deepcopy(self.model)
                    for param in self.server_state.parameters():
                        param.data = torch.zeros_like(param.data)
                except Exception as e:
                    print(f"Warning: Error initializing FedDyn server state for node {num_id}: {e}")
                    self.server_state = None
            
            # Node init for fedadam's server
            if hasattr(args, 'server_method') and args.server_method == 'fedadam' and num_id == -1:
                try:
                    m = copy.deepcopy(self.model)
                    self.zero_weights(m)
                    self.m = m
                    v = copy.deepcopy(self.model)
                    self.zero_weights(v)
                    self.v = v
                except Exception as e:
                    print(f"Warning: Error initializing FedAdam components for server: {e}")
                    self.m = None
                    self.v = None
                    
        except Exception as e:
            print(f"Critical error initializing Node {num_id}: {e}")
            raise

    def zero_weights(self, model):
        """Zero out all model weights"""
        try:
            for n, p in model.named_parameters():
                p.data.zero_()
        except Exception as e:
            print(f"Error zeroing weights: {e}")

    def medical_train_val_split(self, data_tuple, valid_ratio):
        """Split medical Q&A data into train and validation sets"""
        try:
            questions, answers = data_tuple
            
            # Ensure we have data
            if len(questions) == 0 or len(answers) == 0:
                print(f"Warning: Empty data for node {self.num_id}")
                # Return empty datasets
                empty_dataset = MedicalQADataset([], [], self.model.tokenizer)
                empty_loader = DataLoader(empty_dataset, batch_size=1, shuffle=False)
                return empty_loader, empty_loader
            
            # Create indices for splitting
            indices = list(range(len(questions)))
            np.random.shuffle(indices)
            
            # Split indices
            validate_size = int(valid_ratio * len(indices))
            val_indices = indices[:validate_size]
            train_indices = indices[validate_size:]
            
            # Ensure we have at least one sample in each set
            if len(train_indices) == 0:
                train_indices = [0]
            if len(val_indices) == 0:
                val_indices = [0] if len(indices) > 0 else []
            
            # Create datasets
            train_questions = [questions[i] for i in train_indices]
            train_answers = [answers[i] for i in train_indices]
            val_questions = [questions[i] for i in val_indices] if val_indices else [questions[0]]
            val_answers = [answers[i] for i in val_indices] if val_indices else [answers[0]]
            
            # Create datasets with tokenization
            max_length = getattr(self.args, 'max_length', 512)
            train_dataset = MedicalQADataset(train_questions, train_answers, self.model.tokenizer, max_length)
            val_dataset = MedicalQADataset(val_questions, val_answers, self.model.tokenizer, max_length)
            
            # Create data loaders
            batch_size = getattr(self.args, 'batch_size', getattr(self.args, 'batchsize', 4))
            val_batch_size = getattr(self.args, 'validate_batchsize', batch_size)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                num_workers=0,
                shuffle=False
            )
            
            return train_loader, val_loader
            
        except Exception as e:
            print(f"Error in medical_train_val_split for node {self.num_id}: {e}")
            # Return minimal datasets
            try:
                empty_dataset = MedicalQADataset(["What is health?"], ["Health is wellbeing."], self.model.tokenizer)
                empty_loader = DataLoader(empty_dataset, batch_size=1, shuffle=False)
                return empty_loader, empty_loader
            except:
                # Last resort - return None
                return None, None

    def train_val_split(self, idxs, train_set, valid_ratio): 
        """Split training data into train and validation sets (legacy method for image data)"""
        try:
            # Handle case where idxs might be a tuple (for medical data)
            if isinstance(idxs, tuple):
                return self.medical_train_val_split(idxs, valid_ratio)
            
            # Legacy handling for indices
            if hasattr(idxs, 'indices'):
                idxs = idxs.indices
            elif not isinstance(idxs, (list, np.ndarray)):
                idxs = list(range(len(train_set)))
            
            np.random.shuffle(idxs)

            validate_size = int(valid_ratio * len(idxs))
            idxs_test = idxs[:validate_size] if validate_size > 0 else [idxs[0]] if idxs else [0]
            idxs_train = idxs[validate_size:] if validate_size < len(idxs) else idxs

            # Import DatasetSplit - this might not exist, so create a simple version
            try:
                from datasets import DatasetSplit
            except ImportError:
                class DatasetSplit:
                    def __init__(self, dataset, indices):
                        self.dataset = dataset
                        self.indices = indices
                    def __len__(self):
                        return len(self.indices)
                    def __getitem__(self, idx):
                        return self.dataset[self.indices[idx]]

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
                shuffle=True
            )
            
            return train_loader, test_loader
        except Exception as e:
            print(f"Error in train_val_split: {e}")
            # Return empty loaders as fallback
            try:
                from datasets import DatasetSplit
                empty_dataset = DatasetSplit(train_set, [])
            except:
                class DatasetSplit:
                    def __init__(self, dataset, indices):
                        self.dataset = dataset
                        self.indices = indices
                    def __len__(self):
                        return len(self.indices)
                    def __getitem__(self, idx):
                        return self.dataset[self.indices[idx]]
                empty_dataset = DatasetSplit(train_set, [])
            
            empty_loader = DataLoader(empty_dataset, batch_size=1, num_workers=0, shuffle=False)
            return empty_loader, empty_loader

    def train_val_split_forServer(self, idxs, train_set, valid_ratio, num_classes=10):
        """Split data for server with balanced classes (legacy method)"""
        try:
            # Handle medical data
            if isinstance(idxs, tuple):
                return self.medical_train_val_split(idxs, valid_ratio)
            
            # Legacy handling for image data
            if hasattr(idxs, 'indices'):
                idxs = idxs.indices
            elif not isinstance(idxs, (list, np.ndarray)):
                idxs = list(range(min(1000, len(train_set))))  # Default subset
            
            np.random.shuffle(idxs)
            
            validate_size = int(valid_ratio * len(idxs))

            # Generate proxy dataset with balanced classes
            idxs_test = []

            if hasattr(self.args, 'longtail_proxyset') and self.args.longtail_proxyset == 'none':
                test_class_count = [int(validate_size)/num_classes for _ in range(num_classes)]
            elif hasattr(self.args, 'longtail_proxyset') and self.args.longtail_proxyset == 'LT':
                imb_factor = 0.1
                test_class_count = [
                    int(validate_size/num_classes * (imb_factor**(_classes_idx / (num_classes - 1.0)))) 
                    for _classes_idx in range(num_classes)
                ]
            else:
                # Default balanced distribution
                test_class_count = [int(validate_size)/num_classes for _ in range(num_classes)]

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
                    print(f"Warning: Error processing sample {k}: {e}")
                k += 1
                
            # Ensure we have some test data
            if len(idxs_test) == 0:
                idxs_test = idxs[:min(10, len(idxs))]

            idxs_train = [idx for idx in idxs if idx not in idxs_test]
            
            # Ensure we have some training data
            if len(idxs_train) == 0:
                idxs_train = idxs[len(idxs_test):]

            # Import or create DatasetSplit
            try:
                from datasets import DatasetSplit
            except ImportError:
                class DatasetSplit:
                    def __init__(self, dataset, indices):
                        self.dataset = dataset
                        self.indices = indices
                    def __len__(self):
                        return len(self.indices)
                    def __getitem__(self, idx):
                        return self.dataset[self.indices[idx]]

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
                shuffle=True
            )

            return train_loader, test_loader
        except Exception as e:
            print(f"Error in train_val_split_forServer: {e}")
            # Return minimal loaders
            try:
                from datasets import DatasetSplit
                empty_dataset = DatasetSplit(train_set, [])
            except:
                class DatasetSplit:
                    def __init__(self, dataset, indices):
                        self.dataset = dataset
                        self.indices = indices
                    def __len__(self):
                        return len(self.indices)
                    def __getitem__(self, idx):
                        return self.dataset[self.indices[idx]]
                empty_dataset = DatasetSplit(train_set, [0] if len(train_set) > 0 else [])
            
            empty_loader = DataLoader(empty_dataset, batch_size=1, num_workers=0, shuffle=False)
            return empty_loader, empty_loader


# Tools for long-tailed functions (legacy support)
def label_indices2indices(list_label2indices):
    """Convert list of label indices to flat indices list"""
    try:
        indices_res = []
        for indices in list_label2indices:
            indices_res.extend(indices)
        return indices_res
    except Exception as e:
        print(f"Error in label_indices2indices: {e}")
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
        print(f"Error in _get_img_num_per_cls: {e}")
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
                print(f"Warning: Error processing class {_class}: {e}")
                list_clients_indices.append([])
                
        num_list_clients_indices = label_indices2indices(list_clients_indices)
        print('All num_data_train')
        print(len(num_list_clients_indices))

        return img_num_list, list_clients_indices
    except Exception as e:
        print(f"Error in train_long_tail: {e}")
        # Return default values
        default_img_num = [100] * num_classes
        default_indices = [[] for _ in range(num_classes)]
        return default_img_num, default_indices
