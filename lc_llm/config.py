# Configuration parameters for the LC-LLM baseline

# Model configuration
MODEL_NAME = "microsoft/phi-2"
MAX_LENGTH = 2048
BATCH_SIZE = 4  # Adjust based on your GPU memory
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 100
GRADIENT_ACCUMULATION_STEPS = 4
WEIGHT_DECAY = 0.01

# Data configuration
DATA_DIR = "../output_8sbefore_2safter/"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MAX_SAMPLES = None  # Set to a small number for testing, None for full dataset

# Output configuration
OUTPUT_DIR = "./outputs/lc_llm_baseline/"
SAVE_STEPS = 500
EVAL_STEPS = 100

# Prompt configuration
SYSTEM_MESSAGE = """
Role: You are an expert driving prediction model of an autonomous driving system, that can predict the future driving intention and future 4-second driving trajectory for a given target vehicle, avoiding collision with other vehicles and obstacles on the road.

Context:
- Coordinates: Y-axis is perpendicular, and X-axis is parallel to the direction target vehicle is facing. target vehicle's current position is (0,0). Positive values on the y-axis represent the left side of the target vehicle, and negative values on the y-axis represent the right side of the vehicle.

Output:
- Thought:
  - Notable features
  - Potential behaviors
- Final Answer:
  - Intention:
    - 0: Keep lane; 1: Left lane change; 2: Right lane change. The final answer should be one of the three modes.
  - Trajectory (MOST IMPORTANT): 4 points, one every 1 second
    - [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
"""

# LoRA configuration
USE_LORA = True
LORA_RANK = 64  # As specified in the LC-LLM paper
LORA_ALPHA = 16  # As specified in the LC-LLM paper
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_BIAS = "none"  # Can be "none", "all" or "lora_only"
# LoRA configuration
USE_LORA = True
LORA_RANK = 64  # As specified in the LC-LLM paper
LORA_ALPHA = 16  # As specified in the LC-LLM paper
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_BIAS = "none"  # Can be "none", "all" or "lora_only"
asdfsadf