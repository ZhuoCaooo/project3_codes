# Model configuration (matching paper as closely as possible)
MODEL_NAME = "microsoft/phi-2"  # Using Phi-2 instead of Llama-2-13B for simplicity
MAX_LENGTH = 512  # Paper uses 2048 (CRITICAL: was 128 before!)
BATCH_SIZE = 4  # Paper uses 8
LEARNING_RATE = 5e-4  # Paper uses 5e-4 (higher than your 2e-5)
NUM_EPOCHS = 2  # Paper uses 2 epochs
WARMUP_STEPS = 600  # Paper uses 600
GRADIENT_ACCUMULATION_STEPS = 8  # Paper uses 8
WEIGHT_DECAY = 0.01

# Data configuration
# Dataset paths - UPDATE THESE LINES
TRAIN_FILE = "../data/phi2_training_data.json"
TEST_FILE = "../data/phi2_testing_data.json"    
DATA_FILE = "../data/phi2_training_data.json"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Output configuration
OUTPUT_DIR = "./outputs/"
SAVE_STEPS = 50  # Paper uses 50
EVAL_STEPS = 50  # Paper uses 50
LOGGING_STEPS = 10

# LoRA configuration (matching paper exactly)
USE_LORA = True
LORA_RANK = 64  # Paper uses 64 (you had 8)
LORA_ALPHA = 16  # Paper uses 16 (you had 16 - correct!)
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"      # MLP
]
LORA_BIAS = "none"

# Training configuration (matching paper)
USE_FP16 = True  # Paper uses fp16
LOAD_IN_8BIT = False  # Paper doesn't use quantization during training
USE_DEEPSPEED = False  # Simplified for single GPU

# Evaluation configuration
EVAL_BATCH_SIZE = 8
MAX_NEW_TOKENS = 200  # For generation during evaluation
TEMPERATURE = 0.7  # For inference
TOP_P = 0.9

# Metrics configuration (paper evaluation metrics)
INTENTION_CLASSES = ["Keep lane", "Left lane change", "Right lane change"]
EVALUATE_TRAJECTORY = True
TRAJECTORY_HORIZON = 4  # 4 points, 1 second each


# Reproduction settings
REPRODUCE_PAPER = True  # Use paper's exact settings when True
SEED = 42  # For reproducibility

# Paper comparison settings (for reference)
PAPER_MODEL = "Llama-2-13B-chat"  # What paper actually uses
PAPER_DATASET_SIZE = "~3000 samples"  # Approximate
PAPER_PERFORMANCE = {
    "intention_accuracy": 0.971,  # 97.1% from paper
    "trajectory_rmse": {
        "lateral": 0.210,  # Paper results
        "longitudinal": 0.655
    }
}

# Hardware configuration
USE_GPU = True
DEVICE_MAP = "auto"
LOW_CPU_MEM_USAGE = True
