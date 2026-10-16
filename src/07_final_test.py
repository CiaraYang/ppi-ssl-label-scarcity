# Step 7: Frozen One-Shot Test Evaluation (Run Only After the Method Is Final; Never Repeated)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent/'core'))

from figures import final_test_outputs
from final_test import run

if __name__ == '__main__':
    run()
    final_test_outputs()
