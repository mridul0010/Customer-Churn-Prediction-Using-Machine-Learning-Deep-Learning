"""
Entry point for Streamlit Cloud deployment.
This file imports and runs the main app from the app/ folder.
"""

import sys
from pathlib import Path

# Add the app directory to the path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

# Import and run the main app
from app import *  # noqa: F401, F403
