import os

THIS_PATH = os.path.abspath(__file__)
PROJECT_PATH = os.path.dirname(os.path.dirname(THIS_PATH))

# configurations
ENV_PATH = os.path.join(PROJECT_PATH, ".env")
assert os.path.exists(ENV_PATH)

CONFIGURATION_DIR = os.path.join(PROJECT_PATH, "conf")
assert os.path.exists(CONFIGURATION_DIR)

CORRUPTED_CONFIGURATION_DIR = os.path.join(CONFIGURATION_DIR, "corruption")
CORRUPTED_CONFIGURATION_PATH = os.path.join(CORRUPTED_CONFIGURATION_DIR, "conf.yml")

# output
OUTPUT_DIRECTORY = os.path.join(PROJECT_PATH, "outputs")
CORRUPTED_OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "perturbed")
TEST_OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "test")
TEMPORARY_DIRECTORY = os.path.join(PROJECT_PATH, "temp")

# random seed
RANDOM_SEED = 42

# data
DATASET_PATH = "edinburgh-dawg/labelchaos"

# testing algorithms
TEST_NUM_SAMPLES = 100
