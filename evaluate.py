import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.model import precision_at_k, test

p_at_5 = precision_at_k(test, K=5)
print("Precision@k =", p_at_5)
