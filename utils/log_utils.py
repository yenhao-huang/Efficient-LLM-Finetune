import json
import os

def log_best_trail_hyper(best_run, output_dir):
    output_path = os.path.join(output_dir, "best_trial.json")
    with open(output_path, "w") as f:
        json.dump({
            "trial_id": best_run.run_id,
            "score": best_run.objective,
            "hyperparameters": best_run.hyperparameters
        }, f, indent=4)