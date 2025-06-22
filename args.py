import argparse
import os

def args_parser():
    """
    🦙 Medical Federated Learning Argument Parser
    
    ARCHITECTURE:
    - Server: Llama 7B (Powerful central model)
    - Clients: Llama 3B (Efficient distributed models)
    - Task: Medical Question → Answer Generation
    - Data: Medical Q&A pairs from CSV
    """
    parser = argparse.ArgumentParser(
        description="Medical Federated Learning with Llama 7B Server + Llama 3B Clients"
    )
    
    # Medical Data Arguments
    parser.add_argument('--csv_path', type=str, default='medquad.csv',
                        help="Path to medical Q&A CSV file")
    parser.add_argument('--max_length', type=int, default=512,
                        help="Maximum sequence length for text tokenization")
    parser.add_argument('--min_answer_length', type=int, default=10,
                        help="Minimum answer length in words")
    parser.add_argument('--max_answer_length', type=int, default=200,
                        help="Maximum answer length in words")
    
    # Data Distribution
    parser.add_argument('--dataset', type=str, default='medical_qa',
                        help='Dataset type: medical_qa (default), cifar10, cifar100, etc.')
    parser.add_argument('--iid', type=int, default=0,  
                        help='1 for IID, 0 for non-IID (recommended for medical)')
    parser.add_argument('--batchsize', type=int, default=4,  # Reduced for language models
                        help="Batch size for training")
    parser.add_argument('--batch_size', type=int, default=4,  # Alternative name
                        help="Batch size for training (alternative)")
    parser.add_argument('--validate_batchsize', type=int, default=4, 
                        help="Batch size for validation")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3,  # Lower for more specialization
                        help="Dirichlet alpha for non-IID distribution")
    parser.add_argument('--medical_specialization_level', type=float, default=0.7,
                        help="Level of medical specialization (0.0=general, 1.0=highly specialized)")
    
    # System Configuration
    parser.add_argument('--device', type=str, default='0',
                        help="CUDA device: {0, 1, 2, ...} or 'cpu'")
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help="Use CUDA if available")
    parser.add_argument('--mixed_precision', type=bool, default=True,
                        help="Use mixed precision training (fp16)")
    
    # Federated Learning Configuration
    parser.add_argument('--node_num', type=int, default=10,  # Reduced for medical scenario
                        help="Number of hospitals/clients")
    parser.add_argument('--T', type=int, default=50,  # Reduced for faster training
                        help="Number of communication rounds")
    parser.add_argument('--E', type=int, default=5,  # Increased for language models
                        help="Number of local epochs per round")
    parser.add_argument('--select_ratio', type=float, default=1.0,
                        help="Ratio of client selection in each round")
    parser.add_argument('--random_seed', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--exp_name', type=str, default='MedicalFederatedLearning',
                        help="Experiment name for logging")
    
    # Model Configuration - Server: Llama 7B, Clients: Llama 3B
    parser.add_argument('--local_model', type=str, default='llama_3b',
                        help='Local/Client model type: llama_3b (default)')
    parser.add_argument('--server_model', type=str, default='llama_7b',
                        help='Server model type: llama_7b (default)')
    parser.add_argument('--client_model', type=str, default='llama_3b',
                        help='Client model type: llama_3b (default)')
    parser.add_argument('--model_architecture', type=str, default='transformer',
                        help='Model architecture: transformer (default)')
    parser.add_argument('--vocab_size', type=int, default=50257,  # GPT-2 vocab size for compatibility
                        help='Vocabulary size (50257 for GPT-2 compatibility)')
    parser.add_argument('--server_model_size', type=str, default='7b',
                        help='Server model size: 7b (default)')
    parser.add_argument('--client_model_size', type=str, default='3b',
                        help='Client model size: 3b (default)')
    
    # Server Configuration
    parser.add_argument('--server_method', type=str, default='fedawa',
                        help="Server aggregation method: fedavg (default), fedawa, fedprox")
    parser.add_argument('--server_valid_ratio', type=float, default=0.1,
                        help="Ratio of validation set in central server")
    parser.add_argument('--server_epochs', type=int, default=1,
                        help="Optimizer epochs on server")
    parser.add_argument('--server_optimizer', type=str, default='adamw',
                        help="Server optimizer: adamw (default), adam, sgd")
    parser.add_argument('--server_lr', type=float, default=1e-4,
                        help='Server learning rate')
    
    # Client Configuration
    parser.add_argument('--client_method', type=str, default='local_train',
                        help="Client training method: local_train (default), fedprox")
    parser.add_argument('--optimizer', type=str, default='adamw',  # Better for language models
                        help="Client optimizer: adamw (default), adam, sgd")
    parser.add_argument('--client_valid_ratio', type=float, default=0.2,
                        help="Ratio of validation set in clients")  
    parser.add_argument('--lr', type=float, default=5e-5,  # Appropriate for language models
                        help='Client local learning rate')
    parser.add_argument('--local_wd_rate', type=float, default=0.01,
                        help='Client local weight decay rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (if using SGD)')
    parser.add_argument('--mu', type=float, default=0.01,
                        help="Proximal term mu for FedProx")
    parser.add_argument('--lr_decay', type=float, default=0.95,
                        help="Learning rate decay factor per round")
    
    # Language Model Specific Arguments
    parser.add_argument('--gradient_clipping', type=float, default=1.0,
                        help='Gradient clipping max norm for Llama models')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--generation_max_length', type=int, default=200,
                        help='Maximum length for medical text generation')
    parser.add_argument('--generation_temperature', type=float, default=0.7,
                        help='Temperature for text generation (lower = more focused)')
    parser.add_argument('--generation_num_beams', type=int, default=4,
                        help='Number of beams for beam search in generation')
    parser.add_argument('--torch_dtype', type=str, default='float16',
                        help='Model dtype: float16 (default), bfloat16, float32')
    
    # Medical Domain Specific Arguments
    parser.add_argument('--medical_domains', type=str, 
                        default='cardiology,oncology,neurology,endocrinology,general_practice',
                        help='Comma-separated list of medical domains')
    parser.add_argument('--domain_specialization', type=bool, default=True,
                        help='Enable domain-specific client specialization')
    parser.add_argument('--cross_domain_validation', type=bool, default=True,
                        help='Validate models across different medical domains')
    
    # Evaluation Arguments
    parser.add_argument('--eval_metrics', type=str, default='perplexity,bleu,quality_score',
                        help='Comma-separated evaluation metrics')
    parser.add_argument('--eval_frequency', type=int, default=5,
                        help='Evaluate model every N rounds')
    parser.add_argument('--save_model_frequency', type=int, default=10,
                        help='Save model every N rounds')
    parser.add_argument('--generate_samples', type=bool, default=True,
                        help='Generate sample answers during evaluation')
    
    # Logging and Output Arguments
    parser.add_argument('--save_csv', type=bool, default=True,
                        help='Save metrics to CSV files')
    parser.add_argument('--csv_output_dir', type=str, default='./results',
                        help='Directory for CSV output files')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='Logging level: DEBUG, INFO, WARNING, ERROR')
    parser.add_argument('--verbose', type=bool, default=True,
                        help='Enable verbose output')
    
    # Backward Compatibility Arguments (for legacy code)
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes for classification tasks (legacy)')
    parser.add_argument('--image_size', type=int, default=32,
                        help='Image size for vision tasks (legacy)')
    parser.add_argument('--channels', type=int, default=3,
                        help='Number of image channels (legacy)')
    
    # Hardware and Performance Arguments
    parser.add_argument('--num_workers', type=int, default=0,  # 0 to avoid multiprocessing issues
                        help='Number of data loader workers')
    parser.add_argument('--pin_memory', type=bool, default=True,
                        help='Pin memory for faster data transfer')
    
    args = parser.parse_args()
    
    # Post-processing and validation
    args = validate_and_process_args(args)
    
    return args

def validate_and_process_args(args):
    """Validate and process arguments for medical federated learning"""
    
    # Convert string lists to actual lists
    if isinstance(args.medical_domains, str):
        args.medical_domains = [domain.strip() for domain in args.medical_domains.split(',')]
    
    if isinstance(args.eval_metrics, str):
        args.eval_metrics = [metric.strip() for metric in args.eval_metrics.split(',')]
    
    # Ensure proper model configuration for medical tasks
    print("🦙 Validating Medical Federated Learning Configuration:")
    print(f"   Server Model: {args.server_model} ({args.server_model_size})")
    print(f"   Client Model: {args.client_model} ({args.client_model_size})")
    print(f"   Dataset: {args.dataset}")
    
    # Force medical task settings
    if args.dataset == 'medical_qa':
        # Ensure local_model matches client_model
        if args.local_model != args.client_model:
            print(f"   Setting local_model to match client_model: {args.client_model}")
            args.local_model = args.client_model
        
        # Ensure model architecture is transformer
        if args.model_architecture != 'transformer':
            print("   Setting model_architecture to 'transformer' for medical Q&A")
            args.model_architecture = 'transformer'
        
        # Validate batch sizes for language models
        if args.batchsize > 8:
            print(f"   Warning: Large batch size ({args.batchsize}) for language models")
        
        # Ensure batch_size matches batchsize
        if args.batch_size != args.batchsize:
            args.batch_size = args.batchsize
        
        # Validate learning rate
        if args.lr > 1e-3:
            print(f"   Warning: High learning rate ({args.lr}) for language models")
    
    # Validate node count vs select ratio
    if args.select_ratio > 1.0:
        print(f"   Warning: select_ratio ({args.select_ratio}) > 1.0. Setting to 1.0.")
        args.select_ratio = 1.0
    
    # Validate medical specialization parameters
    if args.medical_specialization_level > 1.0:
        args.medical_specialization_level = 1.0
    elif args.medical_specialization_level < 0.0:
        args.medical_specialization_level = 0.0
    
    # Create output directory if it doesn't exist
    if args.save_csv:
        os.makedirs(args.csv_output_dir, exist_ok=True)
        print(f"   Output directory: {args.csv_output_dir}")
    
    # Set derived parameters
    args.selected_clients_per_round = max(1, int(args.node_num * args.select_ratio))
    
    # Validate CSV path
    if not os.path.exists(args.csv_path):
        print(f"   Warning: CSV file {args.csv_path} not found. Will use sample data.")
    
    print(f"   ✅ Configuration validated")
    
    return args

def get_medical_args_preset(preset_name='default'):
    """Get predefined argument presets for different medical scenarios"""
    
    presets = {
        'default': {
            'node_num': 10,
            'T': 50,
            'E': 5,
            'lr': 5e-5,
            'batchsize': 4,
            'medical_specialization_level': 0.7,
            'server_model': 'llama_7b',
            'client_model': 'llama_3b'
        },
        'quick_test': {
            'node_num': 5,
            'T': 50,
            'E': 5,
            'lr': 1e-4,
            'batchsize': 2,
            'medical_specialization_level': 0.5,
            'server_model': 'llama_7b',
            'client_model': 'llama_3b'
        },
        'large_hospital_network': {
            'node_num': 20,
            'T': 100,
            'E': 10,
            'lr': 3e-5,
            'batchsize': 6,
            'medical_specialization_level': 0.8,
            'server_model': 'llama_7b',
            'client_model': 'llama_3b'
        },
        'memory_efficient': {
            'node_num': 10,
            'T': 50,
            'E': 5,
            'lr': 5e-5,
            'batchsize': 2,  # Smaller batch for memory constraints
            'medical_specialization_level': 0.7,
            'server_model': 'llama_7b',
            'client_model': 'llama_3b',
            'torch_dtype': 'float16',
            'gradient_clipping': 0.5
        }
    }
    
    return presets.get(preset_name, presets['default'])

def print_args_summary(args):
    """Print a summary of the current arguments"""
    print("=" * 70)
    print("🦙 MEDICAL FEDERATED LEARNING CONFIGURATION")
    print("   Server: Llama 7B | Clients: Llama 3B")
    print("=" * 70)
    print(f"📊 Dataset: {args.dataset}")
    print(f"📄 Data Source: {args.csv_path}")
    print(f"🏥 Hospitals/Clients: {args.node_num}")
    print(f"🔄 Communication Rounds: {args.T}")
    print(f"📚 Local Epochs per Round: {args.E}")
    print(f"📈 Learning Rate: {args.lr}")
    print(f"📦 Batch Size: {args.batchsize}")
    print("─" * 70)
    print("🦙 MODEL ARCHITECTURE:")
    print(f"   Server Model: {args.server_model} ({args.server_model_size})")
    print(f"   Client Model: {args.client_model} ({args.client_model_size})")
    print(f"   Architecture: {args.model_architecture}")
    print(f"   Max Length: {args.max_length}")
    print(f"   Data Type: {args.torch_dtype}")
    print("─" * 70)
    print("🔄 FEDERATED LEARNING:")
    print(f"   Server Method: {args.server_method}")
    print(f"   Client Method: {args.client_method}")
    print(f"   Optimizer: {args.optimizer}")
    print(f"   Client Selection Ratio: {args.select_ratio}")
    print("─" * 70)
    print("🩺 MEDICAL CONFIGURATION:")
    print(f"   Medical Specialization: {args.medical_specialization_level}")
    print(f"   Medical Domains: {', '.join(args.medical_domains)}")
    print(f"   Generation Max Length: {args.generation_max_length}")
    print("─" * 70)
    print(f"💻 Device: {args.device}")
    print(f"🎲 Random Seed: {args.random_seed}")
    print(f"📁 Output Directory: {args.csv_output_dir}")
    print("=" * 70)

def create_args_from_dict(config_dict):
    """Create args object from dictionary (useful for programmatic setup)"""
    
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Set defaults
    defaults = {
        'dataset': 'medical_qa',
        'csv_path': 'medquad.csv',
        'max_length': 512,
        'node_num': 10,
        'T': 50,
        'E': 5,
        'lr': 5e-5,
        'batchsize': 4,
        'batch_size': 4,
        'validate_batchsize': 4,
        'server_model': 'llama_7b',
        'client_model': 'llama_3b',
        'local_model': 'llama_3b',
        'server_method': 'fedavg',
        'client_method': 'local_train',
        'optimizer': 'adamw',
        'random_seed': 42,
        'save_csv': True,
        'csv_output_dir': './results',
        'device': '0',
        'iid': 0,
        'dirichlet_alpha': 0.3,
        'medical_specialization_level': 0.7,
        'select_ratio': 1.0,
        'server_valid_ratio': 0.1,
        'client_valid_ratio': 0.2,
        'local_wd_rate': 0.01,
        'momentum': 0.9,
        'mu': 0.01,
        'gradient_clipping': 1.0,
        'generation_max_length': 200,
        'generation_temperature': 0.7,
        'generation_num_beams': 4,
        'torch_dtype': 'float16',
        'medical_domains': 'cardiology,oncology,neurology,endocrinology,general_practice',
        'eval_metrics': 'perplexity,bleu,quality_score',
        'model_architecture': 'transformer',
        'vocab_size': 50257,
        'server_model_size': '7b',
        'client_model_size': '3b',
        'num_workers': 0,
        'pin_memory': True,
        'verbose': True
    }
    
    # Update with provided config
    defaults.update(config_dict)
    
    # Create args object
    args = Args(**defaults)
    
    # Validate and process
    args = validate_and_process_args(args)
    
    return args

if __name__ == '__main__':
    # Example usage
    print("Testing Medical Federated Learning Args Parser...")
    
    try:
        # Test default args
        print("\n1. Testing default arguments:")
        args = create_args_from_dict({})
        print_args_summary(args)
        
        # Test preset
        print("\n2. Testing quick_test preset:")
        preset_config = get_medical_args_preset('quick_test')
        args_preset = create_args_from_dict(preset_config)
        print_args_summary(args_preset)
        
        print("\n✅ Args parser tests passed!")
        
    except Exception as e:
        print(f"\n❌ Args parser test failed: {e}")
        import traceback
        traceback.print_exc()
