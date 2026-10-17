# Step 5: Synthetic Checks — Positive Control and PP-SSL Anchor (Synthetic Data Only; No Real Data Is Read)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent/'core'))

from figures import positive_control_outputs, ppssl_outputs
import positive_control
import ppssl_replication

if __name__ == '__main__':
    positive_control.run()      # Pre-declared expectations; results/positive_control.sqlite
    ppssl_replication.run()     # Shoham et al. (NeurIPS 2025) two-group phenomenon; results/ppssl_replication.sqlite
    positive_control_outputs()
    ppssl_outputs()
