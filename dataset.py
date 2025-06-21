import os
import requests
import pandas as pd
import zipfile
import json
from pathlib import Path
import tarfile
import gzip
from io import StringIO
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

##############################################################################
# Integrated Medical Q&A System
# Combines: Dataset Downloading + Model Training/Inference + Federated Learning
##############################################################################

class IntegratedMedicalQA:
    """Complete Medical Q&A System with Data Download and Model Inference"""
    
    def __init__(self, data_dir='./medical_data', model_name='google/flan-t5-small'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        
        # Model components (loaded lazily)
        self.tokenizer = None
        self.model = None
        self.model_loaded = False
        
        print(f"🏥 Initialized Medical Q&A System")
        print(f"   📁 Data directory: {self.data_dir}")
        print(f"   🤖 Model: {self.model_name}")

    ##########################################################################
    # PART 1: DATASET DOWNLOADING AND MANAGEMENT
    ##########################################################################
    
    def download_medquad_dataset(self):
        """Download MedQuAD dataset from GitHub"""
        try:
            print("🏥 Downloading MedQuAD dataset...")
            
            # MedQuAD GitHub repository
            base_url = "https://raw.githubusercontent.com/abachaa/MedQuAD/master"
            
            # Collection of medical Q&A files
            collections = [
                "1_CancerGov_QA/CancerGov_QA.json",
                "2_GARD_QA/GARD_QA.json", 
                "3_GHR_QA/GHR_QA.json",
                "4_MPlusCHC_QA/MPlusCHC_QA.json",
                "5_MPlusHT_QA/MPlusHT_QA.json",
                "6_NINDS_QA/NINDS_QA.json"
            ]
            
            all_qa_pairs = []
            
            for collection in collections:
                url = f"{base_url}/{collection}"
                print(f"   📥 Downloading {collection.split('/')[-1]}...")
                
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    # Parse JSON data
                    data = response.json()
                    
                    # Extract Q&A pairs
                    for item in data.get('documents', []):
                        for qa in item.get('qas', []):
                            question = qa.get('question', '').strip()
                            answer = qa.get('answer', '').strip()
                            
                            if question and answer:
                                all_qa_pairs.append({
                                    'question': question,
                                    'answer': answer,
                                    'source': collection.split('/')[0],
                                    'focus_area': self._extract_focus_area(collection)
                                })
                                
                except Exception as e:
                    print(f"   ⚠️ Failed to download {collection}: {e}")
                    continue
            
            # Save to CSV
            if all_qa_pairs:
                df = pd.DataFrame(all_qa_pairs)
                csv_path = self.data_dir / 'medquad.csv'
                df.to_csv(csv_path, index=False)
                print(f"✅ MedQuAD dataset saved: {csv_path}")
                print(f"   📊 Total Q&A pairs: {len(df)}")
                return str(csv_path)
            else:
                print("❌ No data downloaded from MedQuAD")
                return None
                
        except Exception as e:
            print(f"❌ Error downloading MedQuAD: {e}")
            return None
    
    def create_sample_dataset(self):
        """Create comprehensive sample medical Q&A dataset"""
        try:
            print("🏥 Creating sample medical dataset...")
            
            medical_qa_data = [
                {
                    'question': 'What are the symptoms of diabetes?',
                    'answer': 'Common symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, fatigue, blurred vision, slow-healing wounds, and frequent infections. Type 1 diabetes symptoms often develop quickly, while Type 2 symptoms may develop gradually over years.',
                    'source': 'Medical_Guidelines',
                    'focus_area': 'Endocrinology'
                },
                {
                    'question': 'How is high blood pressure treated?',
                    'answer': 'High blood pressure is treated through lifestyle modifications including regular exercise, healthy diet (low sodium, high potassium), weight management, stress reduction, and limiting alcohol. Medications may include ACE inhibitors, diuretics, beta-blockers, calcium channel blockers, and ARBs.',
                    'source': 'Cardiology_Guidelines', 
                    'focus_area': 'Cardiology'
                },
                {
                    'question': 'What causes heart disease?',
                    'answer': 'Heart disease is caused by multiple factors including high cholesterol, high blood pressure, smoking, diabetes, obesity, family history, sedentary lifestyle, poor diet, excessive alcohol consumption, and stress. Atherosclerosis is a major underlying mechanism.',
                    'source': 'Cardiology_Guidelines',
                    'focus_area': 'Cardiology' 
                },
                {
                    'question': 'What are the side effects of chemotherapy?',
                    'answer': 'Chemotherapy side effects include nausea, vomiting, fatigue, hair loss, increased infection risk, anemia, mouth sores, diarrhea, constipation, neuropathy, cognitive changes, and potential organ damage. Side effects vary by specific drugs and individual patient factors.',
                    'source': 'Oncology_Guidelines',
                    'focus_area': 'Oncology'
                },
                {
                    'question': 'How can I prevent stroke?',
                    'answer': 'Stroke prevention includes controlling blood pressure, maintaining healthy weight, exercising regularly, eating a balanced diet, not smoking, limiting alcohol consumption, managing diabetes, treating heart conditions, and taking prescribed medications as directed.',
                    'source': 'Neurology_Guidelines',
                    'focus_area': 'Neurology'
                },
                {
                    'question': 'What is pneumonia and how is it treated?',
                    'answer': 'Pneumonia is an infection that inflames air sacs in one or both lungs, causing cough with phlegm, fever, chills, and difficulty breathing. Treatment includes antibiotics for bacterial pneumonia, antivirals for viral pneumonia, rest, fluids, and supportive care.',
                    'source': 'Pulmonology_Guidelines',
                    'focus_area': 'Pulmonology'
                },
                {
                    'question': 'What are the symptoms of migraine headaches?',
                    'answer': 'Migraine symptoms include severe throbbing headache usually on one side, nausea, vomiting, sensitivity to light and sound, and sometimes visual disturbances called aura preceding the headache. Attacks can last 4-72 hours.',
                    'source': 'Neurology_Guidelines',
                    'focus_area': 'Neurology'
                },
                {
                    'question': 'How is depression diagnosed and treated?',
                    'answer': 'Depression is diagnosed through clinical interviews, symptom assessment using standardized scales, and ruling out medical causes. Treatment includes psychotherapy (CBT, interpersonal therapy), antidepressant medications (SSRIs, SNRIs), lifestyle changes, and sometimes electroconvulsive therapy.',
                    'source': 'Psychiatry_Guidelines',
                    'focus_area': 'Psychiatry'
                },
                {
                    'question': 'What causes kidney stones?',
                    'answer': 'Kidney stones are caused by dehydration, certain diets high in sodium/oxalate/protein, obesity, digestive diseases, urinary tract infections, certain medications, and genetic factors. Different types include calcium, uric acid, struvite, and cystine stones.',
                    'source': 'Urology_Guidelines',
                    'focus_area': 'Urology'
                },
                {
                    'question': 'How do you manage chronic pain?',
                    'answer': 'Chronic pain management includes medications (NSAIDs, acetaminophen, opioids when appropriate), physical therapy, cognitive behavioral therapy, relaxation techniques, acupuncture, massage, heat/cold therapy, and lifestyle modifications.',
                    'source': 'Pain_Management_Guidelines',
                    'focus_area': 'Pain_Management'
                },
                {
                    'question': 'What are the risk factors for osteoporosis?',
                    'answer': 'Osteoporosis risk factors include advanced age, female gender, menopause, low calcium/vitamin D intake, sedentary lifestyle, smoking, excessive alcohol, certain medications (corticosteroids), family history, and medical conditions affecting bone metabolism.',
                    'source': 'Orthopedics_Guidelines',
                    'focus_area': 'Orthopedics'
                },
                {
                    'question': 'How is asthma treated?',
                    'answer': 'Asthma treatment includes quick-relief medications (bronchodilators like albuterol) for acute symptoms and long-term control medications (inhaled corticosteroids, LABAs) for persistent asthma. Avoiding triggers and having an action plan are essential.',
                    'source': 'Pulmonology_Guidelines',
                    'focus_area': 'Pulmonology'
                },
                {
                    'question': 'What are the symptoms of anxiety disorders?',
                    'answer': 'Anxiety symptoms include excessive worry, restlessness, fatigue, difficulty concentrating, irritability, muscle tension, sleep disturbances, panic attacks, avoidance behaviors, and physical symptoms like rapid heartbeat and sweating.',
                    'source': 'Psychiatry_Guidelines',
                    'focus_area': 'Psychiatry'
                },
                {
                    'question': 'How do you prevent heart attacks?',
                    'answer': 'Heart attack prevention includes controlling cholesterol and blood pressure, not smoking, exercising regularly, maintaining healthy weight, eating a heart-healthy diet, managing diabetes, limiting alcohol, managing stress, and taking prescribed medications.',
                    'source': 'Cardiology_Guidelines',
                    'focus_area': 'Cardiology'
                },
                {
                    'question': 'What is the treatment for arthritis?',
                    'answer': 'Arthritis treatment includes medications (NSAIDs, DMARDs, biologics), physical therapy, occupational therapy, weight management, low-impact exercise, hot/cold therapy, joint injections, and sometimes surgery for severe cases.',
                    'source': 'Rheumatology_Guidelines',
                    'focus_area': 'Rheumatology'
                }
            ]
            
            # Save to CSV
            df = pd.DataFrame(medical_qa_data)
            csv_path = self.data_dir / 'medical_dataset.csv'
            df.to_csv(csv_path, index=False)
            print(f"✅ Sample medical dataset saved: {csv_path}")
            print(f"   📊 Total Q&A pairs: {len(df)}")
            return str(csv_path)
            
        except Exception as e:
            print(f"❌ Error creating sample dataset: {e}")
            return None
    
    def download_pubmed_qa_dataset(self):
        """Download PubMed QA dataset"""
        try:
            print("🏥 Downloading PubMed QA dataset...")
            
            # Try to download from HuggingFace datasets
            try:
                from datasets import load_dataset
                dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train[:1000]")  # First 1000 samples
                
                qa_pairs = []
                for item in dataset:
                    question = item.get('question', '').strip()
                    context = item.get('context', {})
                    answer = item.get('final_decision', '').strip()
                    
                    if question and answer:
                        qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'source': 'PubMed',
                            'focus_area': 'Medical_Research'
                        })
                
                if qa_pairs:
                    df = pd.DataFrame(qa_pairs)
                    csv_path = self.data_dir / 'pubmed_qa.csv'
                    df.to_csv(csv_path, index=False)
                    print(f"✅ PubMed QA dataset saved: {csv_path}")
                    print(f"   📊 Total Q&A pairs: {len(df)}")
                    return str(csv_path)
                    
            except ImportError:
                print("   ⚠️ datasets library not available, using fallback data")
                return self.create_sample_dataset()
                
        except Exception as e:
            print(f"❌ Error downloading PubMed QA: {e}")
            return self.create_sample_dataset()
    
    def _extract_focus_area(self, collection_name):
        """Extract medical focus area from collection name"""
        focus_map = {
            '1_CancerGov_QA': 'Oncology',
            '2_GARD_QA': 'Rare_Diseases', 
            '3_GHR_QA': 'Genetics',
            '4_MPlusCHC_QA': 'General_Health',
            '5_MPlusHT_QA': 'Health_Topics',
            '6_NINDS_QA': 'Neurology'
        }
        return focus_map.get(collection_name, 'General_Medicine')
    
    def ensure_dataset(self, dataset_name='auto'):
        """Ensure medical dataset exists, download if needed"""
        
        print(f"🔍 Ensuring dataset: {dataset_name}")
        
        # Check if dataset already exists
        existing_files = list(self.data_dir.glob('*.csv'))
        if existing_files and dataset_name == 'auto':
            csv_path = existing_files[0]
            print(f"✅ Using existing dataset: {csv_path}")
            return str(csv_path)
        
        # Download based on preference
        if dataset_name in ['auto', 'medquad']:
            csv_path = self.download_medquad_dataset()
            if csv_path:
                return csv_path
        
        if dataset_name in ['auto', 'pubmed']:
            csv_path = self.download_pubmed_qa_dataset()
            if csv_path:
                return csv_path
        
        if dataset_name in ['auto', 'sample', 'fallback']:
            csv_path = self.create_sample_dataset()
            if csv_path:
                return csv_path
        
        raise Exception("Failed to download any medical dataset")

    ##########################################################################
    # PART 2: MODEL LOADING AND INFERENCE
    ##########################################################################
    
    def load_model(self):
        """Load the transformer model for Q&A"""
        if self.model_loaded:
            return
        
        try:
            print(f"🤖 Loading model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model_loaded = True
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def answer_question(self, question):
        """
        Main function: Input question → Output answer
        """
        # Ensure model is loaded
        if not self.model_loaded:
            self.load_model()
        
        # Format the input
        input_text = f"Answer this medical question: {question}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the answer
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    ##########################################################################
    # PART 3: FEDERATED LEARNING SUPPORT
    ##########################################################################
    
    def prepare_federated_data(self, num_nodes=3, csv_path=None):
        """Prepare data for federated learning by splitting by medical specialties"""
        
        if csv_path is None:
            csv_path = self.ensure_dataset()
        
        print(f"🔄 Preparing federated learning data...")
        print(f"   📁 Source: {csv_path}")
        print(f"   🏥 Number of nodes (hospitals): {num_nodes}")
        
        # Load the dataset
        df = pd.read_csv(csv_path)
        
        # Group by focus area (medical specialties)
        specialties = df['focus_area'].unique()
        print(f"   🎯 Available specialties: {', '.join(specialties)}")
        
        # Distribute specialties across nodes
        federated_data = {}
        for i in range(num_nodes):
            node_name = f"Hospital_{chr(65+i)}"  # Hospital_A, Hospital_B, etc.
            federated_data[node_name] = {
                'specialties': [],
                'data': pd.DataFrame(),
                'qa_count': 0
            }
        
        # Round-robin assignment of specialties
        for idx, specialty in enumerate(specialties):
            node_idx = idx % num_nodes
            node_name = f"Hospital_{chr(65+node_idx)}"
            
            specialty_data = df[df['focus_area'] == specialty]
            federated_data[node_name]['specialties'].append(specialty)
            federated_data[node_name]['data'] = pd.concat([
                federated_data[node_name]['data'], 
                specialty_data
            ], ignore_index=True)
            federated_data[node_name]['qa_count'] = len(federated_data[node_name]['data'])
        
        # Save federated datasets
        for node_name, node_data in federated_data.items():
            node_path = self.data_dir / f"{node_name.lower()}_data.csv"
            node_data['data'].to_csv(node_path, index=False)
            
            print(f"   🏥 {node_name}:")
            print(f"      📊 Q&A pairs: {node_data['qa_count']}")
            print(f"      🎯 Specialties: {', '.join(node_data['specialties'])}")
            print(f"      💾 Saved to: {node_path}")
        
        return federated_data

    ##########################################################################
    # PART 4: DEMONSTRATION AND TESTING
    ##########################################################################
    
    def demonstrate_system(self):
        """Complete system demonstration"""
        
        print("=" * 70)
        print("🏥 INTEGRATED MEDICAL Q&A SYSTEM DEMONSTRATION")
        print("=" * 70)
        
        # Step 1: Ensure dataset exists
        print("\n📊 STEP 1: DATA PREPARATION")
        csv_path = self.ensure_dataset('auto')
        
        # Show dataset info
        df = pd.read_csv(csv_path)
        print(f"   ✅ Dataset loaded: {len(df)} Q&A pairs")
        print(f"   🎯 Focus areas: {', '.join(df['focus_area'].unique())}")
        
        # Step 2: Prepare federated data
        print("\n🔄 STEP 2: FEDERATED LEARNING SETUP")
        federated_data = self.prepare_federated_data(num_nodes=3, csv_path=csv_path)
        
        # Step 3: Load model and test Q&A
        print("\n🤖 STEP 3: MODEL INFERENCE TESTING")
        
        # Test questions
        test_questions = [
            "What are the symptoms of diabetes?",
            "How is high blood pressure treated?", 
            "What causes heart disease?",
            "What are the side effects of chemotherapy?"
        ]
        
        print("   Loading model...")
        self.load_model()
        
        print("\n   Testing Q&A system:")
        print("   " + "-" * 60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n   {i}. QUESTION: {question}")
            answer = self.answer_question(question)
            print(f"      ANSWER: {answer}")
        
        print("\n✅ DEMONSTRATION COMPLETE!")
        
    def show_data_flow(self):
        """Show how the federated learning data flows"""
        
        print("\n" + "=" * 70)
        print("🔄 FEDERATED LEARNING DATA FLOW")
        print("=" * 70)
        
        # Sample data structure
        sample_data = {
            'question': [
                "What is diabetes?",
                "How to treat cancer?",
                "What causes heart attack?"
            ],
            'answer': [
                "Diabetes is a group of diseases that result in too much sugar in the blood...",
                "Cancer treatment may include surgery, chemotherapy, radiation therapy...",
                "Heart attacks are usually caused by blocked blood flow to the heart muscle..."
            ],
            'source': ['NIH', 'Mayo Clinic', 'Cleveland Clinic'],
            'focus_area': ['Endocrinology', 'Oncology', 'Cardiology']
        }
        
        df = pd.DataFrame(sample_data)
        
        print("\n1. INPUT DATA STRUCTURE:")
        print("┌─────────────────────────────┬──────────────────────────────┬─────────────────┬──────────────┐")
        print("│ question                    │ answer                       │ source          │ focus_area   │")
        print("├─────────────────────────────┼──────────────────────────────┼─────────────────┼──────────────┤")
        for _, row in df.iterrows():
            q = row['question'][:26] + "..." if len(row['question']) > 26 else row['question']
            a = row['answer'][:27] + "..." if len(row['answer']) > 27 else row['answer']
            print(f"│ {q:<27} │ {a:<28} │ {row['source']:<15} │ {row['focus_area']:<12} │")
        print("└─────────────────────────────┴──────────────────────────────┴─────────────────┴──────────────┘")
        
        print("\n2. FEDERATED DISTRIBUTION:")
        print("   🏥 Hospital A: Gets Endocrinology questions & answers")
        print("   🏥 Hospital B: Gets Oncology questions & answers") 
        print("   🏥 Hospital C: Gets Cardiology questions & answers")
        
        print("\n3. TRAINING PROCESS:")
        print("   📚 Each hospital trains on their specialty data")
        print("   🤖 Model learns: Medical Question → Medical Answer")
        print("   🔒 No patient data shared between hospitals")
        print("   🔄 Only model updates are shared (federated learning)")
        
        print("\n4. FINAL INTEGRATED SYSTEM:")
        print("   📥 Input:  'What are the symptoms of diabetes?'")
        print("   🤖 Processing: Model generates medical answer")
        print("   📤 Output: 'Diabetes symptoms include frequent urination...'")

##############################################################################
# EASY-TO-USE FUNCTIONS
##############################################################################

def create_medical_qa_system(data_dir='./medical_data', model_name='google/flan-t5-small'):
    """
    Create and initialize the integrated medical Q&A system
    
    Args:
        data_dir: Directory to store medical datasets
        model_name: Transformer model for Q&A inference
    
    Returns:
        IntegratedMedicalQA instance
    """
    return IntegratedMedicalQA(data_dir=data_dir, model_name=model_name)

def quick_medical_qa(question, system=None):
    """
    Quick function to get a medical answer
    
    Args:
        question: Medical question to answer
        system: Existing IntegratedMedicalQA instance (optional)
    
    Returns:
        Medical answer string
    """
    if system is None:
        system = create_medical_qa_system()
    
    return system.answer_question(question)

##############################################################################
# MAIN EXECUTION
##############################################################################

if __name__ == "__main__":
    # Create the integrated system
    print("🚀 STARTING INTEGRATED MEDICAL Q&A SYSTEM")
    
    # Initialize system
    medical_qa = create_medical_qa_system()
    
    # Run full demonstration
    medical_qa.demonstrate_system()
    
    # Show data flow
    medical_qa.show_data_flow()
    
    print("\n🎉 SYSTEM READY FOR USE!")
    print("\nQuick usage examples:")
    print("─" * 40)
    print("# Ask a question")
    print("answer = medical_qa.answer_question('What is hypertension?')")
    print("print(answer)")
    print()
    print("# Quick one-liner")
    print("answer = quick_medical_qa('How to prevent diabetes?')")
    print("print(answer)")
    print()
    print("# Prepare federated learning data")
    print("federated_data = medical_qa.prepare_federated_data(num_nodes=5)")
