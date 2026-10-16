# Step 4: Run or Resume the Same-Architecture Teacher and Label-Free Preprocessing Study

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent/'core'))

from teacher_study import main

if __name__ == '__main__':
    main()
