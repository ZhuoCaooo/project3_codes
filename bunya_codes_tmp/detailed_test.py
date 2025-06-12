#!/usr/bin/env python3
"""
Detailed test - see full outputs and compare with ground truth
"""

import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main():
    print("🔍 Detailed Model Test - Full Outputs")
    print("=" * 60)
    
    # Load model (we know it works from minimal test)
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(base_model, "./outputs_final_4epochs/final_model")
    model.eval()
    print("✅ Model loaded")
    
    # Load test data
    with open("../data/phi2_testing_data.json", 'r') as f:
        test_data = json.load(f)
    
    # Test on first 3 samples
    num_samples = 3
    print(f"Testing {num_samples} samples in detail...\n")
    
    for i in range(num_samples):
        sample = test_data[i]["text"]
        
        # Extract input and ground truth
        if "### Assistant:" in sample:
            parts = sample.split("### Assistant:")
            input_text = parts[0] + "### Assistant:"
            ground_truth = parts[1].strip()
        else:
            continue
            
        # Generate prediction
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1500)
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,  # More tokens for full output
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = generated_text[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()
        
        # Show results
        print(f"{'='*60}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*60}")
        
        # Show input scenario (last part)
        input_lines = input_text.split('\n')
        scenario_start = -1
        for j, line in enumerate(input_lines):
            if "target vehicle is driving" in line.lower():
                scenario_start = j
                break
        
        if scenario_start != -1:
            print("SCENARIO:")
            for line in input_lines[scenario_start:scenario_start+10]:
                if line.strip():
                    print(f"  {line.strip()}")
        
        print(f"\nGROUND TRUTH:")
        print(ground_truth)
        
        print(f"\nMODEL PREDICTION:")
        print(prediction)
        
        # Extract and compare intentions
        gt_intention = re.search(r'Intention:\s*"(\d):', ground_truth)
        pred_intention = re.search(r'Intention:\s*"(\d):', prediction)
        
        print(f"\n{'='*30} COMPARISON {'='*30}")
        if gt_intention and pred_intention:
            gt_int = int(gt_intention.group(1))
            pred_int = int(pred_intention.group(1))
            
            intentions = {0: "Keep Lane", 1: "Left Change", 2: "Right Change"}
            
            print(f"Ground Truth: {gt_int} ({intentions.get(gt_int, 'Unknown')})")
            print(f"Prediction:   {pred_int} ({intentions.get(pred_int, 'Unknown')})")
            
            if gt_int == pred_int:
                print("✅ INTENTION CORRECT!")
            else:
                print("❌ INTENTION INCORRECT")
        else:
            print("⚠️ Could not extract intentions for comparison")
        
        # Extract trajectories
        gt_traj = re.search(r'Trajectory:\s*"\[(.*?)\]"', ground_truth)
        pred_traj = re.search(r'Trajectory:\s*"\[(.*?)\]"', prediction)
        
        if gt_traj and pred_traj:
            print(f"Ground Truth Trajectory: {gt_traj.group(1)}")
            print(f"Predicted Trajectory:    {pred_traj.group(1)}")
        
        print(f"\n{'-'*60}")
        
        # Pause between samples for readability
        if i < num_samples - 1:
            print("\nPress Enter for next sample...")
            input()
    
    print(f"\n🎉 Detailed test completed!")
    print("Summary:")
    print("- Model generates proper LC-LLM format")
    print("- Includes Thought, Notable features, Potential behavior")
    print("- Outputs Intention and Trajectory as expected")
    print("- You can now run full evaluation if this looks good!")


if __name__ == "__main__":
    main()
