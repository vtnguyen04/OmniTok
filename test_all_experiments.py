import os
import glob
import subprocess

exp_dir = "configs/experiment"
yaml_files = glob.glob(f"{exp_dir}/**/*.yaml", recursive=True)

successes = []
failures = []

for yaml_path in sorted(yaml_files):
    exp_name = yaml_path.replace("configs/experiment/", "").replace(".yaml", "")
    print(f"\n{'='*50}\nTesting {exp_name}\n{'='*50}")
    
    cmd = [
        "uv", "run", "python", "train.py",
        f"experiment={exp_name}",
        "training.max_steps=1",
        "training.use_wandb=false",
        "training.mixed_precision=bf16",
        "training.data.batch_size=2",
        "data.batch_size=2",
        "data.num_workers=0"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {exp_name} passed")
        successes.append(exp_name)
    else:
        print(f"❌ {exp_name} failed")
        print(result.stderr[-2000:])
        failures.append((exp_name, result.stderr))

print(f"\n\n{'='*50}")
print(f"Results: {len(successes)} passed, {len(failures)} failed")
if failures:
    print("\nFailures:")
    for f, _ in failures:
        print(f"  - {f}")
