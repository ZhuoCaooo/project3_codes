# Configuration parameters for the LC-LLM baseline

import os

# Memory optimization settings
USE_GRADIENT_CHECKPOINTING = True
USE_MEMORY_EFFICIENT_ATTENTION = True
USE_FP16 = False  # Use half precision
MAX_NEW_TOKENS_GENERATION = 50  # Limit token generation

# Model configuration

MODEL_NAME = "microsoft/phi-2"

MAX_LENGTH = 128

BATCH_SIZE = 4  # Adjust based on your GPU memory

LEARNING_RATE = 1e-6  # Reduce from 1e-5



NUM_EPOCHS = 3

WARMUP_STEPS = 100

GRADIENT_ACCUMULATION_STEPS = 8

WEIGHT_DECAY = 0.01



# Data configuration

DATA_DIR = "/scratch/user/uqzcao2/lane_change_pal/data/output_8sbefore_2safter/"

TRAIN_RATIO = 0.8

VAL_RATIO = 0.1

TEST_RATIO = 0.1

MAX_SAMPLES = None  # Set to a small number for testing, None for full dataset



# Output configuration

OUTPUT_DIR = "/scratch/user/uqzcao2/lane_change_pal/outputs/lc_llm_fixed_v2/"

SAVE_STEPS = 500

EVAL_STEPS = 100

GRADIENT_CLIP_NORM = 0.1  # More aggressive clipping


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

LORA_ALPHA = 8  # Reduce from 32
LORA_RANK = 4  # Consider reducing if still unstable

LORA_DROPOUT = 0.05

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

LORA_BIAS = "none"  # Can be "none", "all" or "lora_only"

# Add gradient value clipping
CLIP_GRAD_VALUE = 0.1
# Loss stabilization
LABEL_SMOOTHING = 0.1  # Add label smoothing
LOSS_SCALING = 0.1  # Scale down the loss
