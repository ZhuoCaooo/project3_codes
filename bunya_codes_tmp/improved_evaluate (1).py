#!/usr/bin/env python3
"""
Improved evaluation script matching the new training approach
- Handles ### Assistant: format
- Compatible with 8-bit quantized LoRA models
- Matches LC-LLM paper metrics
"""

import torch
import json
import re
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedLCLLMEvaluator:
    def __init__(self, model_path, base_model="microsoft/phi-2"):
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the trained model with proper 8-bit + LoRA setup"""
        logger.info("Loading model and tokenizer...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # 8-bit quantization config (matching training setup)
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.float16
        )

        # Load base model with same config as training
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()
        logger.info("✅ Model loaded successfully with 8-bit + LoRA")

    def extract_intention_and_trajectory(self, generated_text):
        """Extract intention and trajectory from generated text"""
        
        # Extract intention - updated regex for your format
        intention_patterns = [
            r'Intention:\s*"(\d):', # Format: "1: Left lane change"
            r'Intention:\s*(\d)',   # Format: just number
            r'"(\d):\s*(?:Keep lane|Left lane change|Right lane change)"'  # Alternative format
        ]
        
        intention = -1
        for pattern in intention_patterns:
            intention_match = re.search(pattern, generated_text)
            if intention_match:
                intention = int(intention_match.group(1))
                break

        # Extract trajectory - more robust patterns
        trajectory_patterns = [
            r'Trajectory:\s*"\[(.*?)\]"',  # Your format: "[(x1,y1), ...]"
            r'Trajectory[^:]*:\s*\[(.*?)\]', # Alternative format
            r'\[([-\d.,\s()]+)\]'  # Fallback: any coordinate list
        ]
        
        trajectory = []
        for pattern in trajectory_patterns:
            trajectory_match = re.search(pattern, generated_text)
            if trajectory_match:
                traj_str = trajectory_match.group(1)
                # Parse trajectory points
                points = re.findall(r'\(([-\d.]+),([-\d.]+)\)', traj_str)
                trajectory = [(float(x), float(y)) for x, y in points]
                if len(trajectory) > 0:  # Only break if we found valid points
                    break

        return intention, trajectory

    def extract_ground_truth(self, sample_text):
        """Extract ground truth from sample with new format"""
        # Split on ### Assistant: instead of [/INST]
        parts = sample_text.split("### Assistant:")
        if len(parts) < 2:
            return -1, []

        response = parts[1].strip()
        return self.extract_intention_and_trajectory(response)

    def generate_prediction(self, input_prompt, max_new_tokens=300):
        """Generate prediction for a single input with better parameters"""
        # Tokenize input
        inputs = self.tokenizer(
            input_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1600,  # Leave room for generation
            padding=False
        ).to(self.model.device)

        # Generate with improved parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,  # Balanced creativity
                top_p=0.9,
                top_k=50,         # Add top-k for better control
                repetition_penalty=1.1,  # Prevent repetition
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )

        # Decode only the new part
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        new_text = generated_text[len(input_text):].strip()

        return new_text

    def prepare_input_prompt(self, sample_text):
        """Prepare input prompt from sample with new format"""
        parts = sample_text.split("### Assistant:")
        if len(parts) >= 2:
            return parts[0] + "### Assistant:"
        else:
            logger.warning("Sample doesn't contain ### Assistant: delimiter")
            return sample_text

    def evaluate_dataset(self, test_file, num_samples=None, save_predictions=True):
        """Evaluate on test dataset with detailed logging"""

        # Load test data
        with open(test_file, 'r') as f:
            test_data = json.load(f)

        if num_samples:
            test_data = test_data[:num_samples]

        logger.info(f"Evaluating on {len(test_data)} samples...")

        # Store results
        true_intentions = []
        pred_intentions = []
        true_trajectories = []
        pred_trajectories = []
        predictions = []  # For saving detailed results

        successful_predictions = 0
        parsing_errors = 0

        for i, sample in enumerate(test_data):
            if i % 20 == 0:
                logger.info(f"Processing sample {i + 1}/{len(test_data)}")

            # Extract input and ground truth
            sample_text = sample["text"]
            input_prompt = self.prepare_input_prompt(sample_text)

            # Get ground truth
            true_intention, true_trajectory = self.extract_ground_truth(sample_text)

            # Generate prediction
            try:
                prediction = self.generate_prediction(input_prompt)
                pred_intention, pred_trajectory = self.extract_intention_and_trajectory(prediction)
                
                if pred_intention != -1:
                    successful_predictions += 1
                else:
                    parsing_errors += 1
                    
            except Exception as e:
                logger.warning(f"Error generating prediction for sample {i}: {e}")
                pred_intention, pred_trajectory = -1, []
                parsing_errors += 1

            # Store results
            true_intentions.append(true_intention)
            pred_intentions.append(pred_intention)
            true_trajectories.append(true_trajectory)
            pred_trajectories.append(pred_trajectory)
            
            # Store detailed prediction for analysis
            if save_predictions:
                predictions.append({
                    "sample_id": i,
                    "input": input_prompt,
                    "prediction": prediction,
                    "true_intention": true_intention,
                    "pred_intention": pred_intention,
                    "true_trajectory": true_trajectory,
                    "pred_trajectory": pred_trajectory
                })

        # Save detailed predictions
        if save_predictions:
            with open("detailed_predictions.json", "w") as f:
                json.dump(predictions, f, indent=2)

        logger.info(f"Successful predictions: {successful_predictions}/{len(test_data)}")
        logger.info(f"Parsing errors: {parsing_errors}/{len(test_data)}")

        return self.compute_metrics(true_intentions, pred_intentions,
                                    true_trajectories, pred_trajectories)

    def compute_metrics(self, true_intentions, pred_intentions,
                        true_trajectories, pred_trajectories):
        """Compute evaluation metrics matching the LC-LLM paper"""

        # Intention prediction metrics
        valid_mask = [(t != -1 and p != -1) for t, p in zip(true_intentions, pred_intentions)]
        valid_true = [t for t, mask in zip(true_intentions, valid_mask) if mask]
        valid_pred = [p for p, mask in zip(pred_intentions, valid_mask) if mask]

        if len(valid_true) > 0:
            # Precision, Recall, F1 for each class
            precision, recall, f1, support = precision_recall_fscore_support(
                valid_true, valid_pred, average=None, labels=[0, 1, 2], zero_division=0
            )

            # Macro averages
            macro_precision = np.mean(precision)
            macro_recall = np.mean(recall)
            macro_f1 = np.mean(f1)

            # Overall accuracy
            intention_accuracy = sum([1 for t, p in zip(valid_true, valid_pred) if t == p]) / len(valid_true)
        else:
            precision = recall = f1 = [0, 0, 0]
            macro_precision = macro_recall = macro_f1 = intention_accuracy = 0

        # Trajectory prediction metrics (RMSE)
        lateral_errors = []
        longitudinal_errors = []
        valid_trajectory_count = 0

        for true_traj, pred_traj in zip(true_trajectories, pred_trajectories):
            if len(true_traj) == 4 and len(pred_traj) == 4:
                valid_trajectory_count += 1
                # Calculate RMSE for lateral (Y) and longitudinal (X)
                true_x = [p[0] for p in true_traj]
                true_y = [p[1] for p in true_traj]
                pred_x = [p[0] for p in pred_traj]
                pred_y = [p[1] for p in pred_traj]

                lateral_errors.extend([(ty - py) ** 2 for ty, py in zip(true_y, pred_y)])
                longitudinal_errors.extend([(tx - px) ** 2 for tx, px in zip(true_x, pred_x)])

        lateral_rmse = np.sqrt(np.mean(lateral_errors)) if lateral_errors else float('inf')
        longitudinal_rmse = np.sqrt(np.mean(longitudinal_errors)) if longitudinal_errors else float('inf')

        # Print results in LC-LLM paper format
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS (LC-LLM Paper Format)")
        print("=" * 70)

        print(f"\nIntention Prediction Results:")
        print(f"Overall Accuracy: {intention_accuracy:.1%} ({len(valid_true)} valid samples)")
        print(f"Macro Precision: {macro_precision:.1%}")
        print(f"Macro Recall: {macro_recall:.1%}")
        print(f"Macro F1: {macro_f1:.1%}")

        print(f"\nPer-class Results:")
        classes = ["Keep Lane (0)", "Left Change (1)", "Right Change (2)"]
        for i, cls in enumerate(classes):
            print(f"{cls}: P={precision[i]:.1%}, R={recall[i]:.1%}, F1={f1[i]:.1%}, Support={support[i] if len(valid_true) > 0 else 0}")

        print(f"\nTrajectory Prediction Results:")
        print(f"Valid trajectories: {valid_trajectory_count}/{len(true_trajectories)}")
        print(f"Lateral RMSE: {lateral_rmse:.3f} m")
        print(f"Longitudinal RMSE: {longitudinal_rmse:.3f} m")

        print(f"\nData Quality:")
        print(f"Parsing success rate: {(len(valid_true)/len(true_intentions)):.1%}")

        return {
            "intention_accuracy": intention_accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "lateral_rmse": lateral_rmse,
            "longitudinal_rmse": longitudinal_rmse,
            "per_class_precision": precision,
            "per_class_recall": recall,
            "per_class_f1": f1,
            "valid_samples": len(valid_true),
            "total_samples": len(true_intentions),
            "parsing_success_rate": len(valid_true)/len(true_intentions)
        }

    def create_confusion_matrix(self, test_file, num_samples=None):
        """Create confusion matrix visualization"""
        with open(test_file, 'r') as f:
            test_data = json.load(f)
            
        if num_samples:
            test_data = test_data[:num_samples]
            
        true_intentions = []
        pred_intentions = []
        
        for sample in test_data:
            sample_text = sample["text"]
            input_prompt = self.prepare_input_prompt(sample_text)
            
            true_intention, _ = self.extract_ground_truth(sample_text)
            
            try:
                prediction = self.generate_prediction(input_prompt)
                pred_intention, _ = self.extract_intention_and_trajectory(prediction)
            except:
                pred_intention = -1
                
            true_intentions.append(true_intention)
            pred_intentions.append(pred_intention)

        valid_mask = [(t != -1 and p != -1) for t, p in zip(true_intentions, pred_intentions)]
        valid_true = [t for t, mask in zip(true_intentions, valid_mask) if mask]
        valid_pred = [p for p, mask in zip(pred_intentions, valid_mask) if mask]

        if len(valid_true) > 0:
            cm = confusion_matrix(valid_true, valid_pred, labels=[0, 1, 2])

            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix - Lane Change Intention Prediction\n(Improved Phi-2 Model)')
            plt.colorbar()

            classes = ['Keep Lane (0)', 'Left Change (1)', 'Right Change (2)']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            # Add text annotations
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('improved_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()


def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained LoRA adapter")
    parser.add_argument("--test_file", required=True, help="Test JSON file")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--base_model", default="microsoft/phi-2", help="Base model name")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ImprovedLCLLMEvaluator(args.model_path, args.base_model)

    # Load model
    evaluator.load_model()

    # Evaluate
    results = evaluator.evaluate_dataset(args.test_file, args.num_samples)

    # Create confusion matrix
    evaluator.create_confusion_matrix(args.test_file, args.num_samples)

    # Save results
    with open("improved_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to improved_evaluation_results.json")


if __name__ == "__main__":
    main()