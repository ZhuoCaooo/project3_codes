# Model configuration (MEMORY OPTIMIZED for Bunya)
MODEL_NAME = "microsoft/phi-2"
MAX_LENGTH = 1024  # REDUCED from 512 to save memory
BATCH_SIZE = 1  # REDUCED from 4 to save memory  
LEARNING_RATE = 5e-4  # Keep same
NUM_EPOCHS = 3  # Keep same
WARMUP_STEPS = 300  # REDUCED proportionally (was 600)
GRADIENT_ACCUMULATION_STEPS = 4  # REDUCED from 8 to save memory
WEIGHT_DECAY = 0.01

# Data configuration
TRAIN_FILE = "../data/phi2_training_data.json"
TEST_FILE = "../data/phi2_testing_data.json"    
DATA_FILE = "../data/phi2_training_data.json"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Output configuration
OUTPUT_DIR = "./outputs/"
SAVE_STEPS = 100  # INCREASED to save storage
EVAL_STEPS = 100  # INCREASED to reduce evaluation frequency
LOGGING_STEPS = 10

# LoRA configuration (keeping paper settings)
USE_LORA = True
LORA_RANK = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"      # MLP
]
LORA_BIAS = "none"

# Training configuration (MEMORY OPTIMIZED)
USE_FP16 = True
LOAD_IN_8BIT = False
USE_DEEPSPEED = False
GRADIENT_CHECKPOINTING = True  # NEW: Enable gradient checkpointing

# Evaluation configuration
EVAL_BATCH_SIZE = 1  # REDUCED from 8
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9

# Metrics configuration
INTENTION_CLASSES = ["Keep lane", "Left lane change", "Right lane change"]
EVALUATE_TRAJECTORY = True
TRAJECTORY_HORIZON = 4

# Memory optimization settings
MAX_SPLIT_SIZE_MB = 128  # NEW: Fragment large allocations
EMPTY_CACHE_STEPS = 50   # NEW: Clear cache periodically

# Reproduction settings
REPRODUCE_PAPER = True
SEED = 42

# Paper comparison settings (for reference)
PAPER_MODEL = "Llama-2-13B-chat"
PAPER_DATASET_SIZE = "~3000 samples"
PAPER_PERFORMANCE = {
    "intention_accuracy": 0.971,
    "lateral": 0.210,
    "longitudinal": 0.655
}

# Hardware configuration
USE_GPU = True
DEVICE_MAP = "auto"
LOW_CPU_MEM_USAGE = True