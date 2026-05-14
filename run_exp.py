"""Helper script to run a single experiment with proper logging."""
import sys
import subprocess

if __name__ == "__main__":
    args = sys.argv[1:]
    tag = "_".join(a for a in args if not a.startswith("--"))
    # Redirect stdout+stderr to both console and file
    log_path = f"Results/{tag}.log"
    cmd = [sys.executable, "0_run_train.py"] + args
    print(f"Running: {' '.join(cmd)}")
    with open(log_path, "w") as f:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        f.write(p.stdout)
        print(p.stdout)
    print(f"\nLog saved to: {log_path}")
