# Configuration override for testing
import os
from config import *

# FORCE OVERRIDE - Memory optimization settings
MAX_LENGTH = 64  # Reduced sequence length
BATCH_SIZE = 1   # Minimum batch size  
MAX_SAMPLES = 50  # Very few samples for testing
NUM_EPOCHS = 1   # One epoch only
GRADIENT_ACCUMULATION_STEPS = 1  # No accumulation for testing

# LoRA optimizations
LORA_RANK = 4     # Very small rank
TARGET_MODULES = ["q_proj", "k_proj"]  # Minimal modules

# Output directory
OUTPUT_DIR = "/scratch/user/uqzcao2/lane_change_pal/outputs/lc_llm_test/"

# Print configuration is active
print("TEST CONFIGURATION ACTIVE - USING MINIMAL RESOURCES")