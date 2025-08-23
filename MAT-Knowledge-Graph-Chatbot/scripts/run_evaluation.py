"""
Script to run chatbot evaluation
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation import run_evaluation_suite

if __name__ == "__main__":
    print("Starting MAT Chatbot Evaluation...")
    results = run_evaluation_suite()
    print("Evaluation complete!")