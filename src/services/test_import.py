import sys
import os
print("Python path:", sys.path)
print("Current working directory:", os.getcwd())

try:
    print("Testing import...")
    from src.ace import ScrapingOrchestrator
    print("SUCCESS: ScrapingOrchestrator imported")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
except Exception as e:
    print(f"OTHER ERROR: {e}")

# Also test if the file exists
import os
if os.path.exists('src/ace.py'):
    print("File src/ace.py EXISTS")
else:
    print("File src/ace.py NOT FOUND")