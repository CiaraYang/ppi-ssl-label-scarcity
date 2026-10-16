# Step 6: Run or Resume the Pre-Declared Improved Pipeline (improved_v1; Validation Only)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent/'core'))

from figures import improved_outputs
from improved import run

if __name__ == '__main__':
    run()
    improved_outputs()
