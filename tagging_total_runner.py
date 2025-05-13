import subprocess
import os
import glob

# Search for all .json files within results/definition/USR and extract only the filenames
files = [os.path.basename(file) for file in glob.glob(os.path.join("results", "definition", "USR", "*.json"))]

dataset_names = ["USR"]
# dataset_names = ["USR"]

results_dir = os.path.join("results", "tag")

for json_file in files:
        for dataset in dataset_names:
            data = dataset
            subprocess_list = ["poetry", "run", "python", os.path.join(".", "scripts", "tagging.py"), "--data", dataset, "--setting_name", json_file]
            print(subprocess_list)
            print("==============================================================================================")
            subprocess.run(subprocess_list)
