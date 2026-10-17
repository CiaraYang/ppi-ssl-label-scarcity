# Step 1: Run or Resume the Main Study (3 Mechanisms × 5 Budgets × 5 Seeds; LR and XGBoost Students; Builds the Split If Missing)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent/'core'))

from experiment import main

if __name__ == '__main__':
    main()
