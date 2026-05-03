import os, glob
files = glob.glob("configs/experiment/**/*.yaml", recursive=True)
for f in files:
    with open(f, "r") as file:
        content = file.read()
    if "# @package _global_" not in content:
        content = "# @package _global_\n" + content
        with open(f, "w") as file:
            file.write(content)
        print(f"Added @package _global_ to {f}")
