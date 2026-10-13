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

# Separate DIV2K candidate experiment. These paths never replace baseline data.
DIV2K_DIR = "DIV2K_train_HR"
DIV2K_OUTPUT_DIR = "outputs/div2k"
DIV2K_MODEL_DIR = "models_div2k"
DIV2K_PILOT_IMAGES = 50
DIV2K_SPLIT_SEED = 42
DIV2K_VALIDATION_FRACTION = 0.2
# Exact RBF-SVR is quadratic in the number of feature rows. Keep large runs safe.
DIV2K_MAX_TRAIN_SAMPLES = 10000

# Isolated original-training + DIV2K augmentation experiment.
MIXED_MODEL_DIR = "models_mixed_div2k"
MIXED_OUTPUT_DIR = "outputs/mixed_div2k"
