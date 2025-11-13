import os

# See more: https://pytorch.org/docs/stable/elastic/run.html#environment-variables
RANK = int(os.getenv("LOCAL_RANK", 0))
IS_GLOBAL_ZERO = RANK == 0
NODE_RANK = int(os.getenv("NODE_RANK", 0))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
