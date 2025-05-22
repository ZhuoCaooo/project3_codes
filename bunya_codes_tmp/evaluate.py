#!/usr/bin/env python3
"""
Evaluation script matching LC-LLM paper metrics
"""

import torch
import json
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt


class LCLLMEvaluator:
    def __init__(self, model_path, base_model="microsoft/phi-2"):
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the trained model"""
        print("Loading model and tokenizer...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()

    def extract_intention_and_trajectory(self, generated_text):
        """Extract intention and trajectory from generated text"""

        # Extract intention
        intention_match = re.search(r'Intention:\s*"(\d):', generated_text)
        if intention_match:
            intention = int(intention_match.group(1))
        else:
            intention = -1  # Could not extract

        # Extract trajectory
        trajectory_match = re.search(r'Trajectory:\s*"\[(.*?)\]"', generated_text)
        trajectory = []

        if trajectory_match:
            traj_str = trajectory_match.group(1)
            # Parse trajectory points
            points = re.findall(r'\(([-\d.]+),([-\d.]+)\)', traj_str)
            trajectory = [(float(x), float(y)) for x, y in points]

        return intention, trajectory

    def extract_ground_truth(self, sample_text):
        """Extract ground truth from sample"""
        # Split the sample to get the response part
        parts = sample_text.split("[/INST]")
        if len(parts) < 2:
            return -1, []

        response = parts[1].replace("</s>", "").strip()
        return self.extract_intention_and_trajectory(response)

    def generate_prediction(self, input_prompt, max_new_tokens=200):
        """Generate prediction for a single input"""
        # Tokenize input
        inputs = self.tokenizer(
            input_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1800  # Leave room for generation
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the new part
        input_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        new_text = generated_text[len(input_text):].strip()

        return new_text

    def evaluate_dataset(self, test_file, num_samples=None):
        """Evaluate on test dataset"""

        # Load test data
        with open(test_file, 'r') as f:
            test_data = json.load(f)

        if num_samples:
            test_data = test_data[:num_samples]

        print(f"Evaluating on {len(test_data)} samples...")

        # Store results
        true_intentions = []
        pred_intentions = []
        true_trajectories = []
        pred_trajectories = []

        for i, sample in enumerate(test_data):
            if i % 50 == 0:
                print(f"Processing sample {i + 1}/{len(test_data)}")

            # Extract input and ground truth
            sample_text = sample["text"]
            parts = sample_text.split("[/INST]")
            input_prompt = parts[0] + "[/INST]"

            # Get ground truth
            true_intention, true_trajectory = self.extract_ground_truth(sample_text)

            # Generate prediction
            try:
                prediction = self.generate_prediction(input_prompt)
                pred_intention, pred_trajectory = self.extract_intention_and_trajectory(prediction)
            except Exception as e:
                print(f"Error generating prediction for sample {i}: {e}")
                pred_intention, pred_trajectory = -1, []

            # Store results
            true_intentions.append(true_intention)
            pred_intentions.append(pred_intention)
            true_trajectories.append(true_trajectory)
            pred_trajectories.append(pred_trajectory)

        return self.compute_metrics(true_intentions, pred_intentions,
                                    true_trajectories, pred_trajectories)

    def compute_metrics(self, true_intentions, pred_intentions,
                        true_trajectories, pred_trajectories):
        """Compute evaluation metrics matching the paper"""

        # Intention prediction metrics
        valid_mask = [(t != -1 and p != -1) for t, p in zip(true_intentions, pred_intentions)]
        valid_true = [t for t, mask in zip(true_intentions, valid_mask) if mask]
        valid_pred = [p for p, mask in zip(pred_intentions, valid_mask) if mask]

        if len(valid_true) > 0:
            # Precision, Recall, F1 for each class
            precision, recall, f1, support = precision_recall_fscore_support(
                valid_true, valid_pred, average=None, labels=[0, 1, 2]
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

        for true_traj, pred_traj in zip(true_trajectories, pred_trajectories):
            if len(true_traj) == 4 and len(pred_traj) == 4:
                # Calculate RMSE for lateral (Y) and longitudinal (X)
                true_x = [p[0] for p in true_traj]
                true_y = [p[1] for p in true_traj]
                pred_x = [p[0] for p in pred_traj]
                pred_y = [p[1] for p in pred_traj]

                lateral_errors.extend([(ty - py) ** 2 for ty, py in zip(true_y, pred_y)])
                longitudinal_errors.extend([(tx - px) ** 2 for tx, px in zip(true_x, pred_x)])

        lateral_rmse = np.sqrt(np.mean(lateral_errors)) if lateral_errors else float('inf')
        longitudinal_rmse = np.sqrt(np.mean(longitudinal_errors)) if longitudinal_errors else float('inf')

        # Print results in paper format
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS (LC-LLM Paper Format)")
        print("=" * 60)

        print("\nIntention Prediction Results:")
        print(f"Overall Accuracy: {intention_accuracy:.1%}")
        print(f"Macro Precision: {macro_precision:.1%}")
        print(f"Macro Recall: {macro_recall:.1%}")
        print(f"Macro F1: {macro_f1:.1%}")

        print(f"\nPer-class Results:")
        classes = ["Keep Lane", "Left Change", "Right Change"]
        for i, cls in enumerate(classes):
            print(f"{cls}: P={precision[i]:.1%}, R={recall[i]:.1%}, F1={f1[i]:.1%}")

        print(f"\nTrajectory Prediction Results:")
        print(f"Lateral RMSE: {lateral_rmse:.3f} m")
        print(f"Longitudinal RMSE: {longitudinal_rmse:.3f} m")

        return {
            "intention_accuracy": intention_accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "lateral_rmse": lateral_rmse,
            "longitudinal_rmse": longitudinal_rmse,
            "per_class_precision": precision,
            "per_class_recall": recall,
            "per_class_f1": f1
        }

    def create_confusion_matrix(self, true_intentions, pred_intentions):
        """Create confusion matrix plot"""
        valid_mask = [(t != -1 and p != -1) for t, p in zip(true_intentions, pred_intentions)]
        valid_true = [t for t, mask in zip(true_intentions, valid_mask) if mask]
        valid_pred = [p for p, mask in zip(pred_intentions, valid_mask) if mask]

        if len(valid_true) > 0:
            cm = confusion_matrix(valid_true, valid_pred, labels=[0, 1, 2])

            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix - Lane Change Intention Prediction')
            plt.colorbar()

            classes = ['Keep Lane', 'Left Change', 'Right Change']
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
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
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
    evaluator = LCLLMEvaluator(args.model_path, args.base_model)

    # Load model
    evaluator.load_model()

    # Evaluate
    results = evaluator.evaluate_dataset(args.test_file, args.num_samples)

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    main()