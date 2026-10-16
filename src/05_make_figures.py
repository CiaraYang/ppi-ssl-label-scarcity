# Step 5: Draw All Figures From the Saved Databases and Print the Key Results

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent/'core'))

from figures import main

if __name__ == '__main__':
    main()
