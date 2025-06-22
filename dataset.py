import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

##############################################################################
# Updated Medical Q&A System for Question-Answer Dataset
# Focus: Predict answers given questions using transformer models
##############################################################################

class MedicalQAPredictor:
    """Medical Q&A System for Answer Prediction from Questions"""
    
    def __init__(self, data_dir='/storage/ds_saksham/FedAWA_quan1bit_LLAMA_20June', model_name='google/flan-t5-small', dataset_path='/storage/ds_saksham/FedAWA_quan1bit_LLAMA_20June/medquad_new.csv'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.dataset_path = dataset_path or 'medquad_new.csv'
        
        # Model components (loaded lazily)
        self.tokenizer = None
        self.model = None
        self.model_loaded = False
        
        # Dataset storage
        self.dataset = None
        self.train_data = None
        self.test_data = None
        self.val_data = None
        
        print(f"🏥 Initialized Medical Q&A Predictor")
        print(f"   📁 Data directory: {self.data_dir}")
        print(f"   🤖 Model: {self.model_name}")
        print(f"   📊 Dataset: {self.dataset_path}")

    ##########################################################################
    # PART 1: DATASET LOADING AND PREPROCESSING
    ##########################################################################
    
    def load_dataset(self, csv_path=None):
        """Load the medical Q&A dataset from CSV"""
        try:
            if csv_path is None:
                csv_path = self.dataset_path
            
            print(f"📊 Loading dataset from: {csv_path}")
            
            # Load the CSV file
            df = pd.read_csv(csv_path)
            
            # Validate dataset structure
            if 'question' not in df.columns or 'answer' not in df.columns:
                raise ValueError("Dataset must contain 'question' and 'answer' columns")
            
            # Clean the data
            df = df.dropna(subset=['question', 'answer'])  # Remove rows with missing Q or A
            df['question'] = df['question'].astype(str).str.strip()
            df['answer'] = df['answer'].astype(str).str.strip()
            
            # Remove empty questions or answers
            df = df[(df['question'] != '') & (df['answer'] != '')]
            
            # Add metadata
            df['question_length'] = df['question'].str.len()
            df['answer_length'] = df['answer'].str.len()
            df['qa_id'] = range(len(df))
            
            self.dataset = df
            
            print(f"✅ Dataset loaded successfully!")
            print(f"   📊 Total Q&A pairs: {len(df)}")
            print(f"   📏 Avg question length: {df['question_length'].mean():.1f} chars")
            print(f"   📏 Avg answer length: {df['answer_length'].mean():.1f} chars")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    def split_dataset(self, test_size=0.2, val_size=0.1, random_state=42):
        """Split dataset into train, validation, and test sets"""
        if self.dataset is None:
            self.load_dataset()
        
        print(f"🔄 Splitting dataset...")
        print(f"   📊 Total samples: {len(self.dataset)}")
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            self.dataset, 
            test_size=test_size, 
            random_state=random_state,
            shuffle=True
        )
        
        # Second split: train vs val
        train, val = train_test_split(
            train_val,
            test_size=val_size/(1-test_size),  # Adjust val_size relative to remaining data
            random_state=random_state,
            shuffle=True
        )
        
        self.train_data = train.reset_index(drop=True)
        self.val_data = val.reset_index(drop=True)
        self.test_data = test.reset_index(drop=True)
        
        print(f"   🎯 Train set: {len(self.train_data)} samples ({len(self.train_data)/len(self.dataset)*100:.1f}%)")
        print(f"   🎯 Validation set: {len(self.val_data)} samples ({len(self.val_data)/len(self.dataset)*100:.1f}%)")
        print(f"   🎯 Test set: {len(self.test_data)} samples ({len(self.test_data)/len(self.dataset)*100:.1f}%)")
        
        return self.train_data, self.val_data, self.test_data
    
    def get_dataset_statistics(self):
        """Get comprehensive dataset statistics"""
        if self.dataset is None:
            self.load_dataset()
        
        stats = {
            'total_samples': len(self.dataset),
            'avg_question_length': self.dataset['question_length'].mean(),
            'avg_answer_length': self.dataset['answer_length'].mean(),
            'max_question_length': self.dataset['question_length'].max(),
            'max_answer_length': self.dataset['answer_length'].max(),
            'min_question_length': self.dataset['question_length'].min(),
            'min_answer_length': self.dataset['answer_length'].min()
        }
        
        return stats

    ##########################################################################
    # PART 2: MODEL LOADING AND INFERENCE
    ##########################################################################
    
    def load_model(self):
        """Load the transformer model for Q&A prediction"""
        if self.model_loaded:
            return
        
        try:
            print(f"🤖 Loading model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Check if it's a sequence-to-sequence or causal model
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self.model_type = "seq2seq"
                print("   📝 Model type: Sequence-to-Sequence")
            except:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.model_type = "causal"
                print("   📝 Model type: Causal Language Model")
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_loaded = True
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict_answer(self, question, max_length=256, num_beams=4, temperature=0.7):
        """
        Main prediction function: Given a question, predict the answer
        
        Args:
            question (str): Medical question to answer
            max_length (int): Maximum length of generated answer
            num_beams (int): Number of beams for beam search
            temperature (float): Sampling temperature
        
        Returns:
            str: Predicted answer
        """
        # Ensure model is loaded
        if not self.model_loaded:
            self.load_model()
        
        # Prepare input text
        if self.model_type == "seq2seq":
            input_text = f"Answer this medical question: {question}"
        else:
            input_text = f"Question: {question}\nAnswer:"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Generate answer
        with torch.no_grad():
            if self.model_type == "seq2seq":
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                # Decode the answer
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                # Decode and extract only the new tokens (answer part)
                full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = full_text[len(input_text):].strip()
        
        return answer
    
    def batch_predict(self, questions, batch_size=8):
        """Predict answers for multiple questions"""
        if not self.model_loaded:
            self.load_model()
        
        answers = []
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            batch_answers = []
            
            for question in batch_questions:
                answer = self.predict_answer(question)
                batch_answers.append(answer)
            
            answers.extend(batch_answers)
            print(f"   Processed {min(i+batch_size, len(questions))}/{len(questions)} questions")
        
        return answers

    ##########################################################################
    # PART 3: FEDERATED LEARNING SUPPORT
    ##########################################################################
    
    def prepare_federated_data(self, num_nodes=3, split_method='random'):
        """
        Prepare data for federated learning
        
        Args:
            num_nodes (int): Number of federated nodes (hospitals)
            split_method (str): 'random', 'sequential', or 'question_type'
        """
        if self.dataset is None:
            self.load_dataset()
        
        print(f"🔄 Preparing federated learning data...")
        print(f"   🏥 Number of nodes: {num_nodes}")
        print(f"   📊 Split method: {split_method}")
        print(f"   📈 Total samples: {len(self.dataset)}")
        
        federated_data = {}
        
        if split_method == 'random':
            # Randomly distribute data across nodes
            shuffled_data = self.dataset.sample(frac=1, random_state=42).reset_index(drop=True)
            node_size = len(shuffled_data) // num_nodes
            
            for i in range(num_nodes):
                node_name = f"Hospital_{chr(65+i)}"  # Hospital_A, Hospital_B, etc.
                start_idx = i * node_size
                end_idx = start_idx + node_size if i < num_nodes - 1 else len(shuffled_data)
                
                federated_data[node_name] = {
                    'data': shuffled_data.iloc[start_idx:end_idx].copy(),
                    'qa_count': end_idx - start_idx,
                    'split_method': split_method
                }
        
        elif split_method == 'sequential':
            # Sequentially distribute data
            node_size = len(self.dataset) // num_nodes
            
            for i in range(num_nodes):
                node_name = f"Hospital_{chr(65+i)}"
                start_idx = i * node_size
                end_idx = start_idx + node_size if i < num_nodes - 1 else len(self.dataset)
                
                federated_data[node_name] = {
                    'data': self.dataset.iloc[start_idx:end_idx].copy(),
                    'qa_count': end_idx - start_idx,
                    'split_method': split_method
                }
        
        elif split_method == 'question_type':
            # Split based on question patterns/types
            question_types = self._categorize_questions()
            unique_types = list(question_types.keys())
            
            for i in range(num_nodes):
                node_name = f"Hospital_{chr(65+i)}"
                assigned_types = unique_types[i::num_nodes]  # Round-robin assignment
                
                node_data = pd.DataFrame()
                for q_type in assigned_types:
                    type_data = question_types[q_type]
                    node_data = pd.concat([node_data, type_data], ignore_index=True)
                
                federated_data[node_name] = {
                    'data': node_data,
                    'qa_count': len(node_data),
                    'question_types': assigned_types,
                    'split_method': split_method
                }
        
        # Save federated datasets
        for node_name, node_info in federated_data.items():
            node_path = self.data_dir / f"{node_name.lower()}_data.csv"
            node_info['data'].to_csv(node_path, index=False)
            
            print(f"   🏥 {node_name}:")
            print(f"      📊 Q&A pairs: {node_info['qa_count']}")
            if 'question_types' in node_info:
                print(f"      🎯 Question types: {', '.join(node_info['question_types'])}")
            print(f"      💾 Saved to: {node_path}")
        
        return federated_data
    
    def _categorize_questions(self):
        """Categorize questions by type/pattern"""
        categories = {
            'what_questions': [],
            'how_questions': [],
            'why_questions': [],
            'symptoms_questions': [],
            'treatment_questions': [],
            'causes_questions': [],
            'other_questions': []
        }
        
        for idx, row in self.dataset.iterrows():
            question = row['question'].lower()
            
            if question.startswith('what'):
                categories['what_questions'].append(row)
            elif question.startswith('how'):
                categories['how_questions'].append(row)
            elif question.startswith('why'):
                categories['why_questions'].append(row)
            elif 'symptom' in question:
                categories['symptoms_questions'].append(row)
            elif any(word in question for word in ['treat', 'treatment', 'therapy']):
                categories['treatment_questions'].append(row)
            elif any(word in question for word in ['cause', 'causes', 'caused']):
                categories['causes_questions'].append(row)
            else:
                categories['other_questions'].append(row)
        
        # Convert lists to DataFrames
        for category in categories:
            categories[category] = pd.DataFrame(categories[category])
        
        return categories

    ##########################################################################
    # PART 4: EVALUATION AND TESTING
    ##########################################################################
    
    def evaluate_model(self, test_questions=None, test_answers=None, num_samples=10):
        """Evaluate model performance on test data"""
        if test_questions is None:
            if self.test_data is None:
                self.split_dataset()
            
            # Use random sample from test data
            sample_data = self.test_data.sample(n=min(num_samples, len(self.test_data)))
            test_questions = sample_data['question'].tolist()
            test_answers = sample_data['answer'].tolist()
        
        print(f"🧪 Evaluating model on {len(test_questions)} samples...")
        
        results = []
        for i, (question, true_answer) in enumerate(zip(test_questions, test_answers)):
            predicted_answer = self.predict_answer(question)
            
            result = {
                'question': question,
                'true_answer': true_answer,
                'predicted_answer': predicted_answer,
                'question_length': len(question),
                'true_answer_length': len(true_answer),
                'predicted_answer_length': len(predicted_answer)
            }
            results.append(result)
            
            print(f"\n📋 Sample {i+1}:")
            print(f"   ❓ Question: {question[:100]}...")
            print(f"   ✅ True Answer: {true_answer[:100]}...")
            print(f"   🤖 Predicted: {predicted_answer[:100]}...")
        
        return results
    
    def demonstrate_prediction(self):
        """Demonstrate the answer prediction capability"""
        print("=" * 70)
        print("🏥 MEDICAL Q&A ANSWER PREDICTION DEMONSTRATION")
        print("=" * 70)
        
        # Load dataset and model
        print("\n📊 STEP 1: LOADING DATASET")
        self.load_dataset()
        stats = self.get_dataset_statistics()
        
        print(f"   ✅ Dataset loaded: {stats['total_samples']} Q&A pairs")
        print(f"   📏 Avg question length: {stats['avg_question_length']:.1f} chars")
        print(f"   📏 Avg answer length: {stats['avg_answer_length']:.1f} chars")
        
        print("\n🤖 STEP 2: LOADING MODEL")
        self.load_model()
        
        print("\n🔄 STEP 3: PREPARING FEDERATED DATA")
        federated_data = self.prepare_federated_data(num_nodes=3, split_method='random')
        
        print("\n🧪 STEP 4: TESTING PREDICTION")
        
        # Get sample questions from dataset
        sample_questions = self.dataset['question'].sample(n=5).tolist()
        
        print("   Testing answer prediction on sample questions:")
        print("   " + "-" * 60)
        
        for i, question in enumerate(sample_questions, 1):
            answer = self.predict_answer(question)
            print(f"\n   {i}. QUESTION: {question}")
            print(f"      PREDICTED ANSWER: {answer}")
        
        print("\n✅ DEMONSTRATION COMPLETE!")

##############################################################################
# EASY-TO-USE FUNCTIONS
##############################################################################

def create_medical_predictor(dataset_path='medquad_new.csv', model_name='google/flan-t5-small'):
    """
    Create and initialize the medical Q&A predictor
    
    Args:
        dataset_path: Path to the CSV file with question,answer columns
        model_name: Transformer model for answer prediction
    
    Returns:
        MedicalQAPredictor instance
    """
    return MedicalQAPredictor(dataset_path=dataset_path, model_name=model_name)

def predict_medical_answer(question, predictor=None, dataset_path='medquad_new.csv'):
    """
    Quick function to predict a medical answer
    
    Args:
        question: Medical question to answer
        predictor: Existing MedicalQAPredictor instance (optional)
        dataset_path: Path to dataset (used if predictor is None)
    
    Returns:
        Predicted medical answer string
    """
    if predictor is None:
        predictor = create_medical_predictor(dataset_path=dataset_path)
    
    return predictor.predict_answer(question)

##############################################################################
# MAIN EXECUTION
##############################################################################

if __name__ == "__main__":
    # Create the predictor system
    print("🚀 STARTING MEDICAL Q&A ANSWER PREDICTOR")
    
    # Initialize system with new dataset
    predictor = create_medical_predictor(dataset_path='medquad_new.csv')
    
    # Run full demonstration
    predictor.demonstrate_prediction()
    
    print("\n🎉 SYSTEM READY FOR ANSWER PREDICTION!")
    print("\nQuick usage examples:")
    print("─" * 50)
    print("# Predict answer for a question")
    print("answer = predictor.predict_answer('What are the symptoms of diabetes?')")
    print("print(answer)")
    print()
    print("# Quick one-liner prediction")
    print("answer = predict_medical_answer('How is hypertension treated?')")
    print("print(answer)")
    print()
    print("# Evaluate model performance")
    print("results = predictor.evaluate_model(num_samples=5)")
    print("print(results)")
    print()
    print("# Prepare federated learning data")
    print("federated_data = predictor.prepare_federated_data(num_nodes=4)")
