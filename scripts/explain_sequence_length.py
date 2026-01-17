#!/usr/bin/env python3
"""
Demo: Why Insufficient Samples Lead to Inference Errors

Root Cause: Sequence Length Requirements
"""

import numpy as np

def create_sequences_demo(data, sequence_length=20):
    """
    Simulates the behavior of the create_sequences function
    
    Parameters:
        data: Input data [sample_count,]
        sequence_length: Sequence length (sliding window size)
    
    Returns:
        Sequence array and corresponding target values
    """
    print(f"\n{'='*60}")
    print(f"Input Data: {len(data)} samples")
    print(f"Sequence Length: {sequence_length}")
    print(f"{'='*60}\n")
    
    X, y = [], []
    
    # The critical loop: range(len(data) - sequence_length)
    max_index = len(data) - sequence_length
    print(f"Loop Range: range(0, {max_index})")
    print(f"  → Sequences that can be generated: {max_index}\n")
    
    if max_index <= 0:
        print(f"❌ Error! Cannot generate sequences")
        print(f"   Reason: Sample count({len(data)}) ≤ Sequence length({sequence_length})")
        print(f"   Requirement: Sample count > {sequence_length}")
        return None, None
    
    print(f"✅ Can generate {max_index} sequences\n")
    print("Sequence Construction Process:")
    
    for i in range(max_index):
        # Slice window [i : i+sequence_length]
        sequence = data[i:i + sequence_length]
        target = data[i + sequence_length]
        
        X.append(sequence)
        y.append(target)
        
        if i < 3:  # Only display the first 3 examples
            print(f"  Sequence {i}: data[{i}:{i+sequence_length}] → Target data[{i+sequence_length}]")
            print(f"          Values: {sequence} → {target}")
    
    if max_index > 3:
        print(f"  ... ({max_index - 3} sequences omitted)")
    
    return np.array(X), np.array(y)


def test_cases():
    """Test cases for different sample counts"""
    
    print("\n" + "="*70)
    print("Test Cases: Sequence Generation with Different Sample Counts")
    print("="*70)
    
    sequence_length = 20
    
    # Case 1: Too few samples (10)
    print("\n【Case 1】Insufficient Samples")
    data1 = np.arange(10)
    X1, y1 = create_sequences_demo(data1, sequence_length)
    
    # Case 2: Sample count equals sequence length (20)
    print("\n\n【Case 2】Sample Count = Sequence Length")
    data2 = np.arange(20)
    X2, y2 = create_sequences_demo(data2, sequence_length)
    
    # Case 3: Slightly more samples (25)
    print("\n\n【Case 3】Slightly more samples (25)")
    data3 = np.arange(25)
    X3, y3 = create_sequences_demo(data3, sequence_length)
    
    # Case 4: Sufficient samples (50)
    print("\n\n【Case 4】Sufficient samples (50)")
    data4 = np.arange(50)
    X4, y4 = create_sequences_demo(data4, sequence_length)
    
    # Summary
    print("\n\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"\nGiven Sequence Length = {sequence_length}:\n")
    print(f"  • Samples = 10  → Sequences = 0  ❌ (Too few to generate)")
    print(f"  • Samples = 20  → Sequences = 0  ❌ (Equal, but still cannot generate)")
    print(f"  • Samples = 21  → Sequences = 1  ⚠️  (Barely usable)")
    print(f"  • Samples = 25  → Sequences = 5  ⚠️  (Still low)")
    print(f"  • Samples = 30  → Sequences = 10 ✅ (Usable)")
    print(f"  • Samples = 50  → Sequences = 30 ✅ (Recommended)")
    print(f"\nFormula: Num Sequences = max(0, Num Samples - Sequence Length)")
    print(f"\nRecommendation: Num Samples ≥ {sequence_length + 30} (to generate at least 30 sequences)")


def explain_batch_outputs_error():
    """Explain the 'batch_outputs' error"""
    
    print("\n\n" + "="*70)
    print("Why does the 'batch_outputs' error occur?")
    print("="*70)
    
    print("""
Execution flow when samples are insufficient:

1️⃣  User Input: 20 samples
2️⃣  Preprocessor creates sequences:
    - sequence_length = 20
    - Sequences to generate = 20 - 20 = 0
    - Result: X=[], y=[]  (Empty arrays!)

3️⃣  Model Prediction:
    - model.predict(X) where X.shape = (0, 20, 3)
    - Keras finds no data to predict
    - The loop body never executes
    - The variable 'batch_outputs' is never assigned

4️⃣  Variable Access:
    - Code attempts to access 'batch_outputs'
    - UnboundLocalError: cannot access local variable 'batch_outputs'

Solutions:
  ✅ Ensure Sample Count > sequence_length
  ✅ Recommended: Sample Count ≥ sequence_length + 30
  ✅ For sequence_length=20, at least 50 samples are suggested
""")


if __name__ == '__main__':
    test_cases()
    explain_batch_outputs_error()
    
    print("\n" + "="*70)
    print("Minimum Sample Requirements in Actual Projects")
    print("="*70)
    
    

    print("""
Training Config: sequence_length = 20

Sample requirements for Inference:
  • Minimum: 21 samples (generates 1 sequence)
  • Safe Value: 50 samples (generates 30 sequences) ✅
  • Recommended: 100+ samples (generates 80+ sequences) ✅✅

Why are more samples needed?
  1. Statistical Significance: More samples → More reliable performance evaluation
  2. HTML Reports: Detailed reports are generated for the first 10 samples by default
  3. Distribution Analysis: Enough samples are needed for meaningful state distribution analysis
  4. Visualization: More data points → More meaningful visualizations

Recommendation:
  python scripts/run_inference_uci.py \\
    --model-dir outputs/training/26-01-16/models \\
    --test-data data/uci/splits/test.csv \\
    --n-samples 50   # Or more ✅
""")
    
    print("\n" + "="*70 + "\n")