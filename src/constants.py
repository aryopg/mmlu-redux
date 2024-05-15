import os

PROJECT_PATH = os.path.abspath('./')

# configurations
ENV_PATH = os.path.join(PROJECT_PATH, '.env')
CONFIGURATION_DIR = os.path.join(PROJECT_PATH, 'conf')
CORRUPTED_CONFIGURATION_DIR = os.path.join(CONFIGURATION_DIR, 'corruption')
CORRUPTED_CONFIGURATION_PATH = os.path.join(CORRUPTED_CONFIGURATION_DIR, 'conf.yml')

# output
OUTPUT_DIRECTORY = os.path.join(PROJECT_PATH, 'outputs')
PERTURBED_OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'perturbed')
TEST_OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'test')
TEMPORARY_DIRECTORY = os.path.join(PROJECT_PATH, 'temp')

# random seed
RANDOM_SEED = 42

# data
DATASET_PATH = 'alessiodevoto/purelabel'

# testing algorithms
TEST_NUM_SAMPLES = 100