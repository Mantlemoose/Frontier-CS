# generate_and_run.py
import os
import subprocess

# Test case n values
values = [
    100, 3000, 10000,
    100000, 1000000, 10000000,
    100000000, 1000000000, 10000000000,
    100000000000, 1000000000000
]

os.makedirs("dataset", exist_ok=True)

for i, n in enumerate(values, start=1):
    infile = os.path.join("dataset", f"{i}.in")
    outfile = os.path.join("dataset", f"{i}.ans")

    # Write input file
    with open(infile, "w") as f:
        f.write(f"{n}\n")

    print(f"Generated {infile} with n = {n}")

    # Run ./h < infile > outfile
    try:
        with open(infile, "r") as fin, open(outfile, "w") as fout:
            subprocess.run(["./h"], stdin=fin, stdout=fout, check=True)
        print(f"   -> Output written to {outfile}")
    except subprocess.CalledProcessError as e:
        print(f"   Error running ./h on {infile}: {e}")