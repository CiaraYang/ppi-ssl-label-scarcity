# Step 1: Build the Fixed Train / Validation / Test Split (Only If It Does Not Exist Yet)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent/'core'))

from model_utils import SPLIT_PATH
from pipeline import prepare_split

if __name__ == '__main__':
    existed = SPLIT_PATH.exists()
    prepare_split()
    print(f"Split {'already present' if existed else 'created'}: {SPLIT_PATH}")
