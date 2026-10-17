# Step 2: Run or Resume the Ablations: A1 Replay/Teacher-Only/Fixed λ=1, A2 Rounds, A3 Capacity

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent/'core'))

from ablations import main

if __name__ == '__main__':
    main()
