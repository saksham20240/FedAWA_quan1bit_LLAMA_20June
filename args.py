import argparse

def args_parser():
    """
    🦙 Medical Federated Learning Argument Parser
    
    ARCHITECTURE:
    - Server: Llama 7B (Powerful central model)
    - Clients: Llama 3B (Efficient distributed models)
    - Task: Medical Question-Answering
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
    parser.add_argument('--noniid_type', type=str, default='medical_specialty',
                        help="iid, dirichlet, or medical_specialty")
    parser.add_argument('--iid', type=int, default=0,  
                        help='set 1 for iid, 0 for non-iid (recommended for medical)')
    parser.add_argument('--batchsize', type=int, default=4,  # Reduced for language models
                        help="batch size for training")
    parser.add_argument('--validate_batchsize', type=int, default=4, 
                        help="batch size for validation")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3,  # Lower for more specialization
                    help="dirichlet alpha for non-iid distribution")
    parser.add_argument('--dirichlet_alpha2', type=float, default=False, 
                    help="secondary dirichlet alpha")
    parser.add_argument('--medical_specialization_level', type=float, default=0.7,
                    help="Level of medical specialization (0.0=general, 1.0=highly specialized)")
    parser.add_argument('--longtail_proxyset', type=str, default='none',
                    help="longtail proxy dataset configuration")
    parser.add_argument('--longtail_clients', type=str, default='none', 
                    help="longtail client configuration")
    
    # System Configuration
    parser.add_argument('--device', type=str, default='0',
                        help="CUDA device: {0, 1, 2, ...} or 'cpu'")
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help="Use CUDA if available")
    parser.add_argument('--mixed_precision', type=bool, default=True,
                        help="Use mixed precision training (fp16)")
    
    # Federated Learning Configuration
    parser.add_argument('--node_num', type=int, default=5,  # Reduced for medical scenario
                        help="Number of hospitals/clients")
    parser.add_argument('--T', type=int, default=20,  # Reduced for faster training
                        help="Number of communication rounds")
    parser.add_argument('--E', type=int, default=3,  # Increased for language models
                        help="Number of local epochs per round")
    parser.add_argument('--dataset', type=str, default='medical_qa',
                        help="Dataset type: {medical_qa, cifar10, cifar100, mnist, fmnist}") 
    parser.add_argument('--select_ratio', type=float, default=1.0,
                    help="Ratio of client selection in each round")
    parser.add_argument('--random_seed', type=int, default=42,  # Changed to common ML seed
                        help="Random seed for reproducibility")
    parser.add_argument('--exp_name', type=str, default='MedicalFederatedLearning',
                        help="Experiment name for logging")
    
    # Model Configuration - Server: Llama 7B, Clients: Llama 3B
    parser.add_argument('--local_model', type=str, default='llama_3b',
                        help='Local/Client model type: {llama_3b, gpt2, client, CNN, ResNet20}')
    parser.add_argument('--server_model', type=str, default='llama_7b',
                        help='Server model type: {llama_7b, gpt2-medium, server}')
    parser.add_argument('--client_model', type=str, default='llama_3b',
                        help='Client model type: {llama_3b, gpt2, client}')
    parser.add_argument('--model_architecture', type=str, default='transformer',
                        help='Model architecture: {transformer, cnn, resnet}')
    parser.add_argument('--vocab_size', type=int, default=32000,  # Llama vocab size
                        help='Vocabulary size for Llama models (32000) or GPT2 (50257)')
    parser.add_argument('--server_model_size', type=str, default='7b',
                        help='Server model size: {7b, 13b, 70b}')
    parser.add_argument('--client_model_size', type=str, default='3b',
                        help='Client model size: {1b, 3b, 7b}')
    
    # Server Configuration
    parser.add_argument('--server_method', type=str, default='fedavg',  # Changed to standard FedAvg
                        help="Server aggregation method: {fedavg, fedawa, fedprox, feddyn}")
    parser.add_argument('--server_valid_ratio', type=float, default=0.1,  # Increased for better validation
                    help="Ratio of validation set in central server")
    parser.add_argument('--server_epochs', type=int, default=1,
                        help="Optimizer epochs on server")
    parser.add_argument('--server_optimizer', type=str, default='adamw',  # Better for transformers
                        help="Server optimizer: {adamw, adam, sgd}")
    parser.add_argument('--gamma', type=float, default=1.0,
                        help="Scaling factor for server updates")
    parser.add_argument('--reg_distance', type=str, default='cos',
                        help="Distance metric: {cos, euc}")
    parser.add_argument('--server_lr', type=float, default=1e-4,
                        help='Server learning rate')
                        
    # Client Configuration
    parser.add_argument('--client_method', type=str, default='local_train',
                        help="Client training method: {local_train, fedprox}")
    parser.add_argument('--optimizer', type=str, default='adamw',  # Better for language models
                        help="Client optimizer: {adamw, adam, sgd}")
    parser.add_argument('--client_valid_ratio', type=float, default=0.2,  # Reduced for more training data
                    help="Ratio of validation set in clients")  
    parser.add_argument('--lr', type=float, default=5e-5,  # Much smaller for language models
                        help='Client local learning rate')
    parser.add_argument('--local_wd_rate', type=float, default=0.01,  # Adjusted for transformers
                        help='Client local weight decay rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (if using SGD)')
    parser.add_argument('--mu', type=float, default=0.01,  # Increased for stability
                        help="Proximal term mu for FedProx")
    
    # Language Model Specific Arguments - Optimized for Llama
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
    parser.add_argument('--llama_rope_scaling', type=float, default=1.0,
                        help='RoPE scaling factor for Llama models')
    parser.add_argument('--attention_implementation', type=str, default='sdpa',
                        help='Attention implementation: {sdpa, flash_attention_2, eager}')
    parser.add_argument('--torch_dtype', type=str, default='float16',
                        help='Model dtype: {float16, bfloat16, float32} for memory efficiency')
    
    # Medical Domain Specific Arguments
    parser.add_argument('--medical_domains', type=str, 
                        default='cardiology,oncology,neurology,endocrinology,general_practice',
                        help='Comma-separated list of medical domains')
    parser.add_argument('--domain_specialization', type=bool, default=True,
                        help='Enable domain-specific client specialization')
    parser.add_argument('--cross_domain_validation', type=bool, default=True,
                        help='Validate models across different medical domains')
    parser.add_argument('--medical_terminology_weight', type=float, default=1.5,
                        help='Weight for medical terminology in loss function')
    
    # Evaluation Arguments
    parser.add_argument('--eval_metrics', type=str, default='perplexity,bleu,rouge,semantic_similarity',
                        help='Comma-separated evaluation metrics')
    parser.add_argument('--eval_frequency', type=int, default=5,
                        help='Evaluate model every N rounds')
    parser.add_argument('--save_model_frequency', type=int, default=10,
                        help='Save model every N rounds')
    parser.add_argument('--generate_samples', type=bool, default=True,
                        help='Generate sample answers during evaluation')
    
    # Privacy and Security Arguments
    parser.add_argument('--differential_privacy', type=bool, default=False,
                        help='Enable differential privacy')
    parser.add_argument('--privacy_epsilon', type=float, default=1.0,
                        help='Privacy budget epsilon for differential privacy')
    parser.add_argument('--secure_aggregation', type=bool, default=False,
                        help='Enable secure aggregation protocols')
    
    # Compression and Efficiency Arguments  
    parser.add_argument('--model_compression', type=str, default='none',
                        help='Model compression method: {none, quantization, pruning, distillation}')
    parser.add_argument('--quantization_bits', type=int, default=8,
                        help='Number of bits for quantization')
    parser.add_argument('--communication_compression', type=bool, default=True,
                        help='Compress model updates during communication')
    parser.add_argument('--gradient_compression_ratio', type=float, default=0.1,
                        help='Compression ratio for gradients')
    
    # Logging and Output Arguments
    parser.add_argument('--save_csv', type=bool, default=True,
                        help='Save metrics to CSV files')
    parser.add_argument('--csv_output_dir', type=str, default='./results',
                        help='Directory for CSV output files')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='Logging level: {DEBUG, INFO, WARNING, ERROR}')
    parser.add_argument('--verbose', type=bool, default=True,
                        help='Enable verbose output')
    parser.add_argument('--tensorboard_logging', type=bool, default=False,
                        help='Enable TensorBoard logging')
    
    # Backward Compatibility Arguments (for legacy image classification)
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes for classification tasks')
    parser.add_argument('--image_size', type=int, default=32,
                        help='Image size for vision tasks')
    parser.add_argument('--channels', type=int, default=3,
                        help='Number of image channels')
    
    # Advanced Federated Learning Arguments
    parser.add_argument('--personalization', type=bool, default=False,
                        help='Enable personalized federated learning')
    parser.add_argument('--adaptation_steps', type=int, default=5,
                        help='Number of adaptation steps for personalization')
    parser.add_argument('--meta_learning', type=bool, default=False,
                        help='Enable meta-learning approaches')
    parser.add_argument('--continual_learning', type=bool, default=False,
                        help='Enable continual learning capabilities')
    
    # Hardware and Performance Arguments
    parser.add_argument('--num_workers', type=int, default=0,  # Set to 0 to avoid multiprocessing issues
                        help='Number of data loader workers')
    parser.add_argument('--pin_memory', type=bool, default=True,
                        help='Pin memory for faster data transfer')
    parser.add_argument('--persistent_workers', type=bool, default=False,
                        help='Keep data loader workers persistent')
    parser.add_argument('--compile_model', type=bool, default=False,
                        help='Use torch.compile for faster training (PyTorch 2.0+)')
    
    args = parser.parse_args()
    
    # Post-processing and validation
    args = validate_and_process_args(args)
    
    return args

def validate_and_process_args(args):
    """Validate and process arguments for medical federated learning with Llama models"""
    
    # Convert string lists to actual lists
    if isinstance(args.medical_domains, str):
        args.medical_domains = [domain.strip() for domain in args.medical_domains.split(',')]
    
    if isinstance(args.eval_metrics, str):
        args.eval_metrics = [metric.strip() for metric in args.eval_metrics.split(',')]
    
    # Ensure proper Llama model configuration
    print("🦙 Validating Llama Model Configuration:")
    print(f"   Server Model: {args.server_model} ({args.server_model_size})")
    print(f"   Client Model: {args.client_model} ({args.client_model_size})")
    
    # Validate server model (should be Llama 7B)
    if args.server_model not in ['llama_7b', 'gpt2-medium', 'server']:
        print(f"Warning: Unusual server model '{args.server_model}'. Recommended: 'llama_7b'")
    
    # Validate client model (should be Llama 3B) 
    if args.client_model not in ['llama_3b', 'gpt2', 'client']:
        print(f"Warning: Unusual client model '{args.client_model}'. Recommended: 'llama_3b'")
    
    # Ensure local_model matches client_model for consistency
    if args.local_model != args.client_model:
        print(f"Info: Setting local_model to match client_model: {args.client_model}")
        args.local_model = args.client_model
    
    # Adjust vocab size based on model type
    if 'llama' in args.server_model.lower() or 'llama' in args.client_model.lower():
        if args.vocab_size == 50257:  # GPT2 default
            args.vocab_size = 32000  # Llama default
            print("Info: Updated vocab_size to 32000 for Llama models")
    
    # Ensure batch sizes are reasonable for Llama models
    if args.dataset == 'medical_qa':
        if args.batchsize > 8:
            print(f"Warning: Large batch size ({args.batchsize}) for Llama models. Consider reducing to 2-4.")
        if args.lr > 1e-3:
            print(f"Warning: High learning rate ({args.lr}) for Llama models. Consider using 1e-5 to 1e-4.")
    
    # Set model-specific defaults for legacy compatibility
    if args.dataset == 'medical_qa':
        if args.local_model in ['CNN', 'ResNet20', 'AlexNet', 'LeNet5']:
            print(f"Warning: {args.local_model} not suitable for medical text. Switching to 'llama_3b'.")
            args.local_model = 'llama_3b'
            args.client_model = 'llama_3b'
    
    # Ensure compatibility between models and tasks
    if args.dataset == 'medical_qa' and args.model_architecture != 'transformer':
        print("Warning: Medical Q&A requires transformer architecture. Switching to 'transformer'.")
        args.model_architecture = 'transformer'
    
    # Create output directory if it doesn't exist
    if args.save_csv:
        import os
        os.makedirs(args.csv_output_dir, exist_ok=True)
    
    # Validate node count vs select ratio
    if args.select_ratio > 1.0:
        print(f"Warning: select_ratio ({args.select_ratio}) > 1.0. Setting to 1.0.")
        args.select_ratio = 1.0
    
    # Validate medical specialization parameters
    if args.medical_specialization_level > 1.0:
        args.medical_specialization_level = 1.0
    elif args.medical_specialization_level < 0.0:
        args.medical_specialization_level = 0.0
    
    # Set derived parameters
    args.selected_clients_per_round = max(1, int(args.node_num * args.select_ratio))
    args.total_samples_estimate = args.node_num * 100  # Rough estimate
    
    # Memory optimization for Llama models
    if 'llama' in args.server_model.lower() or 'llama' in args.client_model.lower():
        if args.torch_dtype == 'float32':
            print("Info: Using float16 instead of float32 for memory efficiency with Llama models")
            args.torch_dtype = 'float16'
    
    return args

def get_medical_args_preset(preset_name='default'):
    """Get predefined argument presets for different medical scenarios with Llama models"""
    
    presets = {
        'default': {
            'node_num': 5,
            'T': 20,
            'E': 3,
            'lr': 5e-5,
            'batchsize': 4,
            'medical_specialization_level': 0.7,
            'server_model': 'llama_7b',
            'client_model': 'llama_3b',
            'server_model_size': '7b',
            'client_model_size': '3b'
        },
        'small_test': {
            'node_num': 3,
            'T': 5,
            'E': 2,
            'lr': 1e-4,
            'batchsize': 2,
            'medical_specialization_level': 0.5,
            'server_model': 'llama_7b',
            'client_model': 'llama_3b',
            'server_model_size': '7b',
            'client_model_size': '3b'
        },
        'large_hospital_network': {
            'node_num': 10,
            'T': 50,
            'E': 5,
            'lr': 3e-5,
            'batchsize': 6,
            'medical_specialization_level': 0.8,
            'server_model': 'llama_7b',
            'client_model': 'llama_3b',
            'server_model_size': '7b',
            'client_model_size': '3b'
        },
        'research_consortium': {
            'node_num': 15,
            'T': 100,
            'E': 3,
            'lr': 2e-5,
            'batchsize': 4,
            'medical_specialization_level': 0.9,
            'server_model': 'llama_7b',
            'client_model': 'llama_3b',
            'server_model_size': '7b',
            'client_model_size': '3b'
        },
        'memory_efficient': {
            'node_num': 5,
            'T': 20,
            'E': 3,
            'lr': 5e-5,
            'batchsize': 2,  # Smaller batch for memory constraints
            'medical_specialization_level': 0.7,
            'server_model': 'llama_7b',
            'client_model': 'llama_3b', 
            'server_model_size': '7b',
            'client_model_size': '3b',
            'torch_dtype': 'float16',  # Memory optimization
            'gradient_clipping': 0.5   # More aggressive clipping
        }
    }
    
    return presets.get(preset_name, presets['default'])

def print_args_summary(args):
    """Print a summary of the current arguments with Llama model details"""
    print("=" * 60)
    print("MEDICAL FEDERATED LEARNING CONFIGURATION")
    print("🦙 LLAMA 7B (SERVER) + LLAMA 3B (CLIENTS)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Data Source: {args.csv_path}")
    print(f"Number of Hospitals/Clients: {args.node_num}")
    print(f"Communication Rounds: {args.T}")
    print(f"Local Epochs per Round: {args.E}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batchsize}")
    print("─" * 60)
    print("🏥 MODEL ARCHITECTURE:")
    print(f"   Server Model: {args.server_model} ({args.server_model_size})")
    print(f"   Client Model: {args.client_model} ({args.client_model_size})")
    print(f"   Architecture: {args.model_architecture}")
    print(f"   Vocabulary Size: {args.vocab_size:,}")
    print(f"   Data Type: {args.torch_dtype}")
    print("─" * 60)
    print("🔄 FEDERATED LEARNING:")
    print(f"   Server Method: {args.server_method}")
    print(f"   Client Method: {args.client_method}")
    print(f"   Optimizer: {args.optimizer}")
    print(f"   Client Selection Ratio: {args.select_ratio}")
    print("─" * 60)
    print("🩺 MEDICAL CONFIGURATION:")
    print(f"   Max Text Length: {args.max_length}")
    print(f"   Medical Specialization: {args.medical_specialization_level}")
    print(f"   Medical Domains: {', '.join(args.medical_domains)}")
    print(f"   Generation Max Length: {args.generation_max_length}")
    print("─" * 60)
    print(f"Device: {args.device}")
    print(f"Random Seed: {args.random_seed}")
    print(f"Experiment Name: {args.exp_name}")
    print("=" * 60)

# Example usage function
def example_usage():
    """Example of how to use the args parser with different configurations"""
    
    # Get default arguments
    args = args_parser()
    
    # Print summary
    print_args_summary(args)
    
    # Apply a preset configuration
    preset_config = get_medical_args_preset('large_hospital_network')
    for key, value in preset_config.items():
        setattr(args, key, value)
    
    print("\nAfter applying 'large_hospital_network' preset:")
    print_args_summary(args)

if __name__ == '__main__':
    example_usage()
