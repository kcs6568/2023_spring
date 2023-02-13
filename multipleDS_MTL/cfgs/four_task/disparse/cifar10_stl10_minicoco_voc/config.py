'''
Configuration File Used for Cityscapes Training & Evaluation
'''

DATA_ROOT = "/root/data/img_type_datasets/cityscapes"
CROP_H = 224
CROP_W = 224
TASKS = ["seg", "depth"]
TASKS_NUM_CLASS = [19, 1]

LAMBDAS = [1, 20]
NUM_GPUS = 1
BATCH_SIZE = 16
# MAX_ITERS = 20000
MAX_ITERS = 100
DECAY_LR_FREQ = 4000
DECAY_LR_RATE = 0.5
    
INIT_LR = 1e-4
WEIGHT_DECAY = 5e-4
IMAGE_SHAPE = (256, 512)

PRUNE_TIMES = 11
PRUNE_ITERS = [100] * PRUNE_TIMES

END = 15000
INT = 50
PRUNE_RATE = 0.5
RETRAIN_EPOCH = 1000
RETRAIN_LR = 1e-5