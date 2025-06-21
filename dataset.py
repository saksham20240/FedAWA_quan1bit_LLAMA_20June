from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd

class SimpleMedicalQA:
    def __init__(self, model_name='google/flan-t5-small'):
        """Initialize the medical Q&A system"""
        print("Loading medical Q&A model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Model loaded successfully!")
    
    def answer_question(self, question):
        """
        Main function: Input question → Output answer
        """
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

def demonstrate_qa_system():
    """Show how the system works with real examples"""
    
    print("=== MEDICAL Q&A SYSTEM DEMONSTRATION ===\n")
    
    # Initialize the Q&A system
    qa_system = SimpleMedicalQA()
    
    # Example medical questions
    medical_questions = [
        "What are the symptoms of diabetes?",
        "How is high blood pressure treated?", 
        "What causes heart disease?",
        "What are the side effects of chemotherapy?",
        "How can I prevent stroke?"
    ]
    
    print("Testing the Q&A system:\n")
    print("FORMAT: Input Question → Output Answer")
    print("-" * 60)
    
    for i, question in enumerate(medical_questions, 1):
        print(f"\n{i}. QUESTION: {question}")
        
        # Get the answer
        answer = qa_system.answer_question(question)
        
        print(f"   ANSWER: {answer}")
        print("-" * 60)

def show_data_flow():
    """Show how the federated learning data flows"""
    
    print("\n=== DATA FLOW IN FEDERATED MEDICAL Q&A ===\n")
    
    # Sample CSV data structure
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
        'focus_area': ['Diabetes', 'Cancer', 'Heart Disease']
    }
    
    df = pd.DataFrame(sample_data)
    
    print("1. INPUT DATA (medquad.csv):")
    print("┌────────────────────────────┬─────────────────────────────┬─────────────────┬──────────────┐")
    print("│ question                   │ answer                      │ source          │ focus_area   │")
    print("├────────────────────────────┼─────────────────────────────┼─────────────────┼──────────────┤")
    for _, row in df.iterrows():
        q = row['question'][:25] + "..." if len(row['question']) > 25 else row['question']
        a = row['answer'][:25] + "..." if len(row['answer']) > 25 else row['answer']
        print(f"│ {q:<26} │ {a:<27} │ {row['source']:<15} │ {row['focus_area']:<12} │")
    print("└────────────────────────────┴─────────────────────────────┴─────────────────┴──────────────┘")
    
    print("\n2. FEDERATED LEARNING PROCESS:")
    print("   Hospital A: Gets diabetes questions & answers")
    print("   Hospital B: Gets cancer questions & answers") 
    print("   Hospital C: Gets heart disease questions & answers")
    
    print("\n3. TRAINING:")
    print("   Each hospital trains: Question → Answer")
    print("   Model learns to generate medical answers")
    print("   No patient data shared between hospitals")
    
    print("\n4. FINAL SYSTEM:")
    print("   Input:  'What are the symptoms of diabetes?'")
    print("   Output: 'Diabetes symptoms include frequent urination...'")

def test_with_your_data():
    """Show how to use with your actual CSV file"""
    
    print("\n=== USING WITH YOUR ACTUAL DATA ===\n")
    
    print("To use with your medquad.csv file:")
    print("""
    # 1. Load your data
    df = pd.read_csv('medquad.csv')
    
    # 2. Set up federated learning
    class Args:
        node_num = 3          # Number of hospitals
        iid = 0               # Non-IID distribution
        dirichlet_alpha = 0.5 # How specialized each hospital is
        model_name = 'google/flan-t5-small'
        max_length = 512
        random_seed = 42
    
    # 3. Create the federated data loader
    from your_updated_code import MedicalQAData
    data_loader = Medical
