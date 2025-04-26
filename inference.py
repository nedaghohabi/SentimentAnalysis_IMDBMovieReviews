# train.py

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Master training script")
    parser.add_argument("train_strategy", type=str, help="Trainer to run (e.g. logreg, llm)")
    args, remaining_args = parser.parse_known_args()

    train_strategy = args.train_strategy
    if train_strategy=="ml_bow":
        trainer_script = f"testers/ml_bagofwords.py"
    elif train_strategy=="ml_wv":
        trainer_script = f"testers/ml_wordvectors.py"
    elif train_strategy=="fcn_bow":
        trainer_script = f"testers/fcn_bagofwords.py"
    elif train_strategy=="fcn_wv":
        trainer_script = f"testers/fcn_wordvectors.py"
    elif train_strategy=="lstm_wv":
        trainer_script = f"testers/lstm_wordvectors.py"
    elif train_strategy=="transformer_wv":
        trainer_script = f"testers/transformer_wordvectors.py"
    elif train_strategy=="llm":
        trainer_script = f"testers/llm.py"
    else:
        raise ValueError("Tester not found")
    
    if not os.path.exists(trainer_script):
        print(f"Tester script '{trainer_script}' not found.")
        sys.exit(1)

    # Call the trainer script and pass all remaining CLI args to it
    cmd = [sys.executable, trainer_script] + remaining_args
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
