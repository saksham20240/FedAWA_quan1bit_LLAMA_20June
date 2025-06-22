import torch
import os
import subprocess

##############################################################################
# CUDA Debugging Tools for Federated Learning
##############################################################################

def enable_cuda_debugging():
    """Enable detailed CUDA error reporting"""
    print("🔧 Enabling CUDA debugging...")
    
    # Enable device-side assertions
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    # Enable CUDA launch blocking for better error traces
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Enable CUDA memory debugging
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    print("   ✅ CUDA_LAUNCH_BLOCKING=1 (synchronous execution)")
    print("   ✅ TORCH_USE_CUDA_DSA=1 (device-side assertions)")
    print("   ✅ Memory debugging enabled")
    
    return True

def check_cuda_environment():
    """Comprehensive CUDA environment check"""
    print("🔍 CUDA Environment Check")
    print("=" * 40)
    
    # Basic CUDA check
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA not available - running on CPU")
        return False
    
    # GPU information
    gpu_count = torch.cuda.device_count()
    current_gpu = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu)
    
    print(f"GPU Count: {gpu_count}")
    print(f"Current GPU: {current_gpu}")
    print(f"GPU Name: {gpu_name}")
    
    # Memory information
    memory_allocated = torch.cuda.memory_allocated() / 1e9
    memory_reserved = torch.cuda.memory_reserved() / 1e9
    memory_total = torch.cuda.get_device_properties(current_gpu).total_memory / 1e9
    
    print(f"Memory Allocated: {memory_allocated:.2f} GB")
    print(f"Memory Reserved: {memory_reserved:.2f} GB") 
    print(f"Memory Total: {memory_total:.2f} GB")
    
    # PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version (PyTorch): {torch.version.cuda}")
    
    # Check CUDA toolkit version
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            nvcc_version = result.stdout.split('release ')[-1].split(',')[0]
            print(f"CUDA Toolkit Version: {nvcc_version}")
        else:
            print("CUDA Toolkit: Not found in PATH")
    except:
        print("CUDA Toolkit: Unable to check version")
    
    return True

def test_basic_cuda_operations():
    """Test basic CUDA operations for debugging"""
    print("\n🧪 Testing Basic CUDA Operations")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    try:
        # Test basic tensor operations
        print("1. Testing tensor creation...")
        x = torch.randn(10, 10).cuda()
        print("   ✅ Tensor creation successful")
        
        print("2. Testing basic arithmetic...")
        y = x + x
        print("   ✅ Basic arithmetic successful")
        
        print("3. Testing matrix multiplication...")
        z = torch.mm(x, y)
        print("   ✅ Matrix multiplication successful")
        
        print("4. Testing GPU-CPU transfer...")
        z_cpu = z.cpu()
        z_gpu = z_cpu.cuda()
        print("   ✅ GPU-CPU transfer successful")
        
        print("5. Testing gradient computation...")
        x.requires_grad_(True)
        loss = (x ** 2).sum()
        loss.backward()
        print("   ✅ Gradient computation successful")
        
        # Cleanup
        del x, y, z, z_cpu, z_gpu, loss
        torch.cuda.empty_cache()
        
        print("✅ All basic CUDA operations successful")
        return True
        
    except Exception as e:
        print(f"❌ CUDA operation failed: {e}")
        return False

def debug_tokenizer_issues(tokenizer, sample_texts=None):
    """Debug common tokenizer issues that cause CUDA errors"""
    print("\n🔤 Debugging Tokenizer Issues")
    print("-" * 40)
    
    if sample_texts is None:
        sample_texts = [
            "What are the symptoms of diabetes?",
            "How is high blood pressure treated?",
            ""  # Empty string test
        ]
    
    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Check special tokens
    special_tokens = {
        'pad_token': tokenizer.pad_token,
        'eos_token': tokenizer.eos_token,
        'bos_token': getattr(tokenizer, 'bos_token', None),
        'unk_token': getattr(tokenizer, 'unk_token', None),
        'sep_token': getattr(tokenizer, 'sep_token', None),
        'cls_token': getattr(tokenizer, 'cls_token', None)
    }
    
    print("\nSpecial Tokens:")
    for name, token in special_tokens.items():
        if token is not None:
            token_id = getattr(tokenizer, f'{name}_id', None)
            print(f"  {name}: '{token}' (ID: {token_id})")
        else:
            print(f"  {name}: None")
    
    # Validate special token IDs
    vocab_size = len(tokenizer)
    print(f"\nValidating Token IDs (vocab_size = {vocab_size}):")
    
    for name, token in special_tokens.items():
        if token is not None:
            token_id = getattr(tokenizer, f'{name}_id', None)
            if token_id is not None:
                if token_id < 0:
                    print(f"  ❌ {name}_id is negative: {token_id}")
                elif token_id >= vocab_size:
                    print(f"  ❌ {name}_id >= vocab_size: {token_id} >= {vocab_size}")
                else:
                    print(f"  ✅ {name}_id is valid: {token_id}")
    
    # Test tokenization
    print("\nTesting Tokenization:")
    for i, text in enumerate(sample_texts):
        try:
            # Test basic tokenization
            tokens = tokenizer(text, return_tensors='pt')
            input_ids = tokens['input_ids']
            
            # Check for invalid token IDs
            invalid_ids = input_ids[input_ids >= vocab_size]
            negative_ids = input_ids[input_ids < 0]
            
            print(f"  Text {i}: '{text[:50]}...'")
            print(f"    Length: {len(text)}")
            print(f"    Token count: {input_ids.shape[1]}")
            print(f"    Max token ID: {input_ids.max().item()}")
            print(f"    Min token ID: {input_ids.min().item()}")
            
            if len(invalid_ids) > 0:
                print(f"    ❌ Invalid token IDs: {invalid_ids.tolist()}")
            else:
                print(f"    ✅ All token IDs valid")
            
            if len(negative_ids) > 0:
                print(f"    ❌ Negative token IDs: {negative_ids.tolist()}")
            else:
                print(f"    ✅ No negative token IDs")
            
        except Exception as e:
            print(f"  ❌ Tokenization failed for text {i}: {e}")
    
    return True

def debug_model_forward_pass(model, tokenizer, device='cuda'):
    """Debug model forward pass issues"""
    print("\n🤖 Debugging Model Forward Pass")
    print("-" * 40)
    
    if not torch.cuda.is_available() and device == 'cuda':
        print("❌ CUDA not available, switching to CPU")
        device = 'cpu'
    
    try:
        model.eval()
        model = model.to(device)
        
        # Test with simple input
        test_input = "What is diabetes?"
        print(f"Testing with: '{test_input}'")
        
        # Tokenize
        inputs = tokenizer(
            test_input,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"Input shape: {inputs['input_ids'].shape}")
        print(f"Input device: {inputs['input_ids'].device}")
        print(f"Max token ID: {inputs['input_ids'].max().item()}")
        print(f"Min token ID: {inputs['input_ids'].min().item()}")
        
        # Test forward pass
        with torch.no_grad():
            if hasattr(model, 'generate'):
                # For generation models
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                print("✅ Generation successful")
                
                # Decode output
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Generated: '{decoded}'")
            else:
                # For other models
                outputs = model(**inputs)
                print("✅ Forward pass successful")
                print(f"Output keys: {outputs.keys()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_debug():
    """Run comprehensive debugging suite"""
    print("🐛 COMPREHENSIVE CUDA DEBUGGING SUITE")
    print("=" * 60)
    
    # Enable debugging
    enable_cuda_debugging()
    
    # Check environment
    cuda_ok = check_cuda_environment()
    
    if cuda_ok:
        # Test basic operations
        ops_ok = test_basic_cuda_operations()
        
        if ops_ok:
            print("\n✅ Basic CUDA operations working")
            
            # Test with a simple model
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                
                print("\n🤖 Testing with T5-small model...")
                tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
                model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
                
                # Debug tokenizer
                debug_tokenizer_issues(tokenizer)
                
                # Debug model
                debug_model_forward_pass(model, tokenizer)
                
            except Exception as e:
                print(f"❌ Model testing failed: {e}")
        
        print("\n🎯 DEBUGGING COMPLETE")
        print("Check the output above for any ❌ errors that need fixing.")
    
    else:
        print("❌ CUDA not available - please check your GPU setup")

# Quick debugging functions
def quick_cuda_check():
    """Quick CUDA availability check"""
    print("🚀 Quick CUDA Check")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def clear_cuda_cache():
    """Clear CUDA cache and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ CUDA cache cleared")
    
    import gc
    gc.collect()
    print("✅ Garbage collection completed")

if __name__ == "__main__":
    # Run the comprehensive debugging suite
    run_comprehensive_debug()
