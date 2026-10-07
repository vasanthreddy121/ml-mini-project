# Project configuration

IMAGE_WIDTH = 500

# SLIC
N_SEGMENTS = 200
COMPACTNESS = 10

# Local feature extraction
PATCH_SIZE = 10

# SVR
SVR_C = 0.125
SVR_EPSILON = 0.0625
SVR_KERNEL = "rbf"

# MRF / ICM
MRF_GAMMA = 2.0
ICM_ITERATIONS = 10

# Paths
TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"
TEST_DIR = "dataset/test"

MODEL_DIR = "models"

OUTPUT_DIR = "outputs"

FEATURE_DIR = "outputs/features"
SEGMENTATION_DIR = "outputs/segmentation"
PROCESSED_DIR = "outputs/processed"
