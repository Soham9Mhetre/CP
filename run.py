"""
Entry point: python run.py
Starts the Crypto Fraud Prevention dashboard on http://localhost:5000
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.api import app

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Crypto Fraud Prevention Dashboard")
    print("  http://localhost:5000")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
