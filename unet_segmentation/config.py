import os
from datetime import datetime

DATASET_NAME = "mnist_extended"
PARENT_DIR = os.path.dirname(__file__)
NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32
PIN_MEMORY = True
SEED = 8
NUM_CLASSES = 5
TRAIN_VAL_SPLIT = 0.9

LR_RATE = 0.001
EPOCHS = 10

MODEL_LOG = "unet_test"

LOAD_MODEL = False
MODEL_PATH = "model.pth"

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")

# ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ['one', 'two', 'three', 'four', 'five']