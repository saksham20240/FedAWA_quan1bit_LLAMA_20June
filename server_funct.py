import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import gc
import os
from pathlib import Path

# Import our custom modules
from your_dataset_file import MedicalQAData
from server_client_update_llama7b import (
    Server_update, Client_update, Client_validate,
    get_memory_usage, get_tensor_memory_usage, calculate_model_size
)
from onebit_medical_qa_federated import (
    MedicalQAClientNode, run_llama7b_medical_qa_federated,
    test_federated_llama7b_medical_qa
)

##############################################################################
# Main Federated Learning Orchestrator
##############################################################################

class LlamaFederatedMedicalQA:
    """
    Complete federated learning system for medical Q&A using Llama 7B
    
    Features:
    - Multi-hospital collaboration
    - Privacy-preserving training
    - Medical specialty specialization
    - Memory-optimized for large models
    """
    
    def __init__(self, config):
        self.config = config
        self.hospitals = []
        self.central_server = None
        self.training_history = []
        
        print("🦙 Initializing Llama 7B Medical Q&A Federated Learning System")
        print("=" * 70)
        self._setup_system()
    
    def _setup_system(self):
        """Initialize the federated learning system"""
        
        # Create results directory
        os.makedirs(self.config.save_path, exist_ok=True)
        
        # Setup medical data distribution
        print("📊 Setting up medical data distribution...")
        self.medical_data = MedicalQAData(self.config)
        
        print(f"   ✅ Loaded {len(self.medical_data.df)} medical Q&A pairs")
        print(f"   🏥 Distributing across {self.config.num_hospitals} hospitals")
        
        # Initialize hospitals
        self._initialize_hospitals()
        
        # Initialize central server
        self._initialize_central_server()
    
    def _initialize_hospitals(self):
        """Initialize hospital client nodes"""
        
        print("🏥 Initializing hospitals with Llama 7B...")
        
        for hospital_id in range(self.config.num_hospitals):
            try:
                print(f"   🏥 Setting up Hospital {hospital_id}...")
                
                # Create hospital node
                hospital = MedicalQAClientNode(
                    client_id=hospital_id,
                    model_name=self.config.model_name,
                    use_4bit=self.config.use_4bit_quantization
                )
                
                # Assign medical data
                hospital_data = self.medical_data.get_client_dataloader(
                    client_id=hospital_id,
                    batch_size=self.config.batch_size,
                    shuffle=True
                )
                hospital.set_local_data(hospital_data)
                
                specialty = hospital.medical_specialties[hospital_id % 5]
                data_size = len(hospital_data.dataset)
                
                print(f"     📋 Specialty: {specialty}")
                print(f"     📊 Data: {data_size} Q&A pairs")
                
                self.hospitals.append(hospital)
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"   ❌ Error initializing Hospital {hospital_id}: {e}")
                continue
        
        print(f"   ✅ Successfully initialized {len(self.hospitals)} hospitals")
    
    def _initialize_central_server(self):
        """Initialize central server"""
        
        print("🌐 Initializing central server...")
        
        try:
            self.central_server = MedicalQAClientNode(
                client_id=-1,  # Server ID
                model_name=self.config.model_name,
                use_4bit=self.config.use_4bit_quantization
            )
            print("   ✅ Central server initialized with Llama 7B")
            
        except Exception as e:
            print(f"   ❌ Error initializing central server: {e}")
            raise
    
    def run_federated_training(self):
        """Execute the complete federated learning process"""
        
        print(f"\n🚀 Starting Federated Training")
        print(f"   🏥 Hospitals: {len(self.hospitals)}")
        print(f"   🔄 Rounds: {self.config.num_rounds}")
        print(f"   🦙 Model: {self.config.model_name}")
        print(f"   📊 Aggregation: {self.config.server_method}")
        
        start_time = time.time()
        
        for round_num in range(1, self.config.num_rounds + 1):
            print(f"\n{'='*60}")
            print(f"📋 ROUND {round_num}/{self.config.num_rounds}")
            print(f"{'='*60}")
            
            round_metrics = self._execute_round(round_num)
            self.training_history.append(round_metrics)
            
            # Save progress
            self._save_round_results(round_num, round_metrics)
            
            # Memory management
            self._cleanup_memory()
        
        total_time = time.time() - start_time
        print(f"\n🎉 Federated Training Complete!")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        
        # Final evaluation and save
        self._final_evaluation()
        self._save_complete_results()
    
    def _execute_round(self, round_num):
        """Execute a single federated learning round"""
        
        round_start = time.time()
        
        # 1. Client Selection (all hospitals for now)
        selected_hospitals = list(range(len(self.hospitals)))
        size_weights = [1.0] * len(self.hospitals)
        
        print(f"🎯 Selected hospitals: {selected_hospitals}")
        
        # 2. Client Training
        print("🏋️ Hospital Training Phase...")
        hospitals, avg_loss = Client_update(self.config, self.hospitals, self.central_server)
        self.hospitals = hospitals
        
        # 3. Server Aggregation
        print("🌐 Server Aggregation Phase...")
        self.central_server = Server_update(
            self.config, 
            self.central_server, 
            self.hospitals, 
            selected_hospitals, 
            size_weights, 
            round_num
        )
        
        # 4. Validation
        print("🧪 Validation Phase...")
        avg_score, hospital_scores = Client_validate(self.config, self.hospitals)
        
        # 5. Collect metrics
        round_metrics = self._collect_round_metrics(
            round_num, avg_loss, avg_score, hospital_scores, round_start
        )
        
        return round_metrics
    
    def _collect_round_metrics(self, round_num, avg_loss, avg_score, hospital_scores, round_start):
        """Collect comprehensive metrics for the round"""
        
        round_time = time.time() - round_start
        
        metrics = {
            'round': round_num,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'avg_training_loss': round(avg_loss, 4),
            'avg_quality_score': round(avg_score, 2),
            'round_time_minutes': round(round_time / 60, 2),
            'num_hospitals': len(self.hospitals),
            'model_name': self.config.model_name,
            'aggregation_method': self.config.server_method,
            'memory_usage_gb': round(get_memory_usage() / 1024, 2),
            'gpu_memory_gb': round(get_tensor_memory_usage() / 1024, 2),
            'hospital_scores': hospital_scores
        }
        
        # Hospital-specific metrics
        hospital_metrics = []
        for i, hospital in enumerate(self.hospitals):
            specialty = hospital.medical_specialties[i % 5]
            data_size = len(hospital.local_data.dataset) if hospital.local_data else 0
            
            hospital_metrics.append({
                'hospital_id': i,
                'specialty': specialty,
                'data_size': data_size,
                'quality_score': hospital_scores[i] if i < len(hospital_scores) else 0,
                'model_size_gb': round(calculate_model_size(hospital.model) / 1024, 2)
            })
        
        metrics['hospitals'] = hospital_metrics
        
        print(f"📊 Round {round_num} Summary:")
        print(f"   💰 Avg Loss: {avg_loss:.4f}")
        print(f"   🎯 Avg Quality: {avg_score:.2f}")
        print(f"   ⏱️ Time: {round_time/60:.1f} min")
        print(f"   💾 Memory: {metrics['memory_usage_gb']} GB")
        
        return metrics
    
    def _save_round_results(self, round_num, metrics):
        """Save round results to CSV"""
        
        try:
            # Convert to DataFrame format
            round_data = []
            for hospital_data in metrics['hospitals']:
                row = {
                    'Round': round_num,
                    'Hospital_ID': hospital_data['hospital_id'],
                    'Medical_Specialty': hospital_data['specialty'],
                    'Data_Size': hospital_data['data_size'],
                    'Quality_Score': hospital_data['quality_score'],
                    'Model_Size_GB': hospital_data['model_size_gb'],
                    'Avg_Training_Loss': metrics['avg_training_loss'],
                    'Round_Time_Minutes': metrics['round_time_minutes'],
                    'Memory_Usage_GB': metrics['memory_usage_gb'],
                    'GPU_Memory_GB': metrics['gpu_memory_gb'],
                    'Model_Name': metrics['model_name'],
                    'Aggregation_Method': metrics['aggregation_method'],
                    'Timestamp': metrics['timestamp']
                }
                round_data.append(row)
            
            df = pd.DataFrame(round_data)
            filename = f"{self.config.save_path}/round_{round_num}_results.csv"
            df.to_csv(filename, index=False)
            
            print(f"💾 Round {round_num} results saved to {filename}")
            
        except Exception as e:
            print(f"⚠️ Error saving round results: {e}")
    
    def _save_complete_results(self):
        """Save complete training history"""
        
        try:
            # Aggregate all rounds
            all_data = []
            for metrics in self.training_history:
                for hospital_data in metrics['hospitals']:
                    row = {
                        'Round': metrics['round'],
                        'Hospital_ID': hospital_data['hospital_id'],
                        'Medical_Specialty': hospital_data['specialty'],
                        'Data_Size': hospital_data['data_size'],
                        'Quality_Score': hospital_data['quality_score'],
                        'Model_Size_GB': hospital_data['model_size_gb'],
                        'Avg_Training_Loss': metrics['avg_training_loss'],
                        'Round_Time_Minutes': metrics['round_time_minutes'],
                        'Memory_Usage_GB': metrics['memory_usage_gb'],
                        'GPU_Memory_GB': metrics['gpu_memory_gb'],
                        'Model_Name': metrics['model_name'],
                        'Aggregation_Method': metrics['aggregation_method'],
                        'Timestamp': metrics['timestamp']
                    }
                    all_data.append(row)
            
            complete_df = pd.DataFrame(all_data)
            filename = f"{self.config.save_path}/complete_training_results.csv"
            complete_df.to_csv(filename, index=False)
            
            print(f"📊 Complete results saved to {filename}")
            
            # Save summary statistics
            self._save_summary_stats(complete_df)
            
        except Exception as e:
            print(f"⚠️ Error saving complete results: {e}")
    
    def _save_summary_stats(self, df):
        """Save training summary statistics"""
        
        try:
            summary = {
                'Training_Summary': {
                    'Total_Rounds': self.config.num_rounds,
                    'Total_Hospitals': len(self.hospitals),
                    'Model_Used': self.config.model_name,
                    'Aggregation_Method': self.config.server_method,
                    'Final_Avg_Quality_Score': df.groupby('Round')['Quality_Score'].mean().iloc[-1],
                    'Final_Avg_Loss': df.groupby('Round')['Avg_Training_Loss'].mean().iloc[-1],
                    'Total_Training_Time_Hours': df.groupby('Round')['Round_Time_Minutes'].first().sum() / 60,
                    'Peak_Memory_Usage_GB': df['Memory_Usage_GB'].max(),
                    'Peak_GPU_Memory_GB': df['GPU_Memory_GB'].max()
                },
                'Hospital_Performance': {}
            }
            
            # Per-hospital summary
            for specialty in df['Medical_Specialty'].unique():
                specialty_data = df[df['Medical_Specialty'] == specialty]
                final_round_data = specialty_data[specialty_data['Round'] == self.config.num_rounds]
                
                summary['Hospital_Performance'][specialty] = {
                    'Final_Quality_Score': final_round_data['Quality_Score'].mean(),
                    'Quality_Improvement': (
                        final_round_data['Quality_Score'].mean() - 
                        specialty_data[specialty_data['Round'] == 1]['Quality_Score'].mean()
                    ),
                    'Total_Data_Size': specialty_data['Data_Size'].iloc[0],
                    'Avg_Model_Size_GB': specialty_data['Model_Size_GB'].mean()
                }
            
            # Save as JSON for easy reading
            import json
            summary_file = f"{self.config.save_path}/training_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"📋 Training summary saved to {summary_file}")
            
        except Exception as e:
            print(f"⚠️ Error saving summary: {e}")
    
    def _final_evaluation(self):
        """Perform final evaluation of the global model"""
        
        print("\n🎯 Final Model Evaluation")
        print("-" * 40)
        
        test_federated_llama7b_medical_qa(self.central_server)
    
    def _cleanup_memory(self):
        """Clean up memory after each round"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

##############################################################################
# Configuration Class
##############################################################################

class FederatedConfig:
    """Configuration for federated learning system"""
    
    def __init__(self):
        # Model settings
        self.model_name = 'meta-llama/Llama-2-7b-chat-hf'
        self.use_4bit_quantization = True
        
        # Federated learning settings
        self.num_hospitals = 3
        self.num_rounds = 3
        self.server_method = 'fedawa'  # 'fedawa' or 'fedavg'
        
        # Training settings
        self.batch_size = 1  # Small for Llama 7B
        self.E = 1  # Local epochs
        self.max_length = 2048
        
        # Data settings
        self.iid = 0  # Non-IID distribution
        self.dirichlet_alpha = 0.2  # High specialization
        self.random_seed = 42
        
        # System settings
        self.use_onebit_training = False
        self.save_path = "llama7b_federated_results/"
        
        # Create Args object for compatibility
        self.node_num = self.num_hospitals

##############################################################################
# Main Execution
##############################################################################

def main():
    """Main execution function"""
    
    print("🦙 Llama 7B Medical Q&A Federated Learning System")
    print("=" * 60)
    print("🏥 Privacy-preserving medical AI training across hospitals")
    print("🤝 Collaborative learning without sharing patient data")
    print("🧠 Advanced question-answering with Llama 7B")
    print("=" * 60)
    
    # Check system requirements
    check_system_requirements()
    
    # Initialize configuration
    config = FederatedConfig()
    
    # Create and run federated learning system
    try:
        fl_system = LlamaFederatedMedicalQA(config)
        fl_system.run_federated_training()
        
        print("\n🎉 SUCCESS: Federated learning completed!")
        print(f"📊 Results saved to: {config.save_path}")
        print("💡 The global model now combines medical knowledge from all hospitals")
        print("🔒 Patient privacy was preserved throughout the process")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check your setup and try again.")

def check_system_requirements():
    """Check if system meets requirements"""
    
    print("🔍 Checking system requirements...")
    
    # Check PyTorch
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("   ❌ PyTorch not found. Please install PyTorch.")
        return False
    
    # Check transformers
    try:
        import transformers
        print(f"   ✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("   ❌ Transformers not found. Please install transformers.")
        return False
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   ✅ CUDA available: {gpu_memory:.1f} GB GPU memory")
        if gpu_memory < 12:
            print("   ⚠️ Warning: Less than 12GB GPU memory. Consider using 4-bit quantization.")
    else:
        print("   ⚠️ CUDA not available. Training will be slow on CPU.")
    
    # Check medical dataset
    if os.path.exists('medquad.csv'):
        print("   ✅ Medical Q&A dataset found")
    else:
        print("   ❌ medquad.csv not found. Please place your medical dataset in the current directory.")
        return False
    
    print("   ✅ System requirements check passed!")
    return True

def quick_demo():
    """Quick demonstration with minimal resources"""
    
    print("🚀 Quick Demo Mode")
    print("-" * 30)
    
    config = FederatedConfig()
    config.num_hospitals = 2
    config.num_rounds = 1
    config.model_name = 'meta-llama/Llama-2-7b-chat-hf'
    config.use_4bit_quantization = True
    config.save_path = "demo_results/"
    
    try:
        fl_system = LlamaFederatedMedicalQA(config)
        fl_system.run_federated_training()
        print("🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        quick_demo()
    else:
        main()
