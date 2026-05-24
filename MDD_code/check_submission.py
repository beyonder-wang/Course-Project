"""
check_submission.py - Verify MDD.txt format is correct for submission.
"""
import sys
sys.path.insert(0, '.')
import h5py
import os


def main():
    errors = []
    warnings = []

    # 1. Check MDD.txt exists
    if not os.path.exists('MDD.txt'):
        print("ERROR: MDD.txt does not exist!")
        return

    # 2. Read MDD.txt
    with open('MDD.txt', 'r') as f:
        lines = f.readlines()

    # 3. Get expected number of samples from test file
    with h5py.File('test_x_only.h5', 'r') as f:
        n_test = f['X'].shape[0]

    print(f"MDD.txt lines: {len(lines)}")
    print(f"Test samples:  {n_test}")

    # 4. Check line count
    if len(lines) != n_test:
        errors.append(f"Line count mismatch: MDD.txt has {len(lines)} lines, test has {n_test} samples")

    # 5. Check each line
    predictions = []
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check for empty lines
        if stripped == '':
            errors.append(f"Line {i+1}: empty line")
            continue

        # Check it's only 0 or 1
        if stripped not in ('0', '1'):
            errors.append(f"Line {i+1}: invalid value '{stripped}' (expected 0 or 1)")
            continue

        # Check for extra whitespace or characters
        if line.rstrip('\n') != stripped:
            warnings.append(f"Line {i+1}: has trailing whitespace")

        predictions.append(int(stripped))

    # 6. Statistics
    if predictions:
        n_0 = predictions.count(0)
        n_1 = predictions.count(1)
        print(f"\nPrediction distribution:")
        print(f"  Class 0 (Healthy Controls):       {n_0} ({n_0/len(predictions)*100:.1f}%)")
        print(f"  Class 1 (Major Depressive Disorder): {n_1} ({n_1/len(predictions)*100:.1f}%)")

        # Sanity check: if all same class, something is likely wrong
        if n_0 == 0 or n_1 == 0:
            warnings.append("All predictions are the same class - this is suspicious!")

        # Check for reasonable distribution (warn if very imbalanced)
        ratio = min(n_0, n_1) / max(n_0, n_1)
        if ratio < 0.2:
            warnings.append(f"Very imbalanced predictions (ratio={ratio:.2f}) - verify this is expected")

    # 7. Report
    print()
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  [X] {e}")
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [!] {w}")

    if not errors:
        print("Submission format OK")
        print(f"   - {len(predictions)} predictions")
        print(f"   - All values are 0 or 1")
        print(f"   - Line count matches test samples ({n_test})")
    else:
        print("\n[X] Submission has errors - please fix before submitting!")

    return len(errors) == 0


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
