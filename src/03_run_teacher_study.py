# Step 3: Same-Architecture Teachers, Label-Free Preprocessing and the Pretrained TabPFN Teacher Pilot

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent/'core'))

try:  # PyTorch (TabPFN Teacher) must load its OpenMP runtime before XGBoost's on macOS, or TabPFN segfaults.
    import torch  # noqa: F401
except ImportError:
    pass

from teacher_study import main

if __name__ == '__main__':
    main()
