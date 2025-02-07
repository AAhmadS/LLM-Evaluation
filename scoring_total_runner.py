import subprocess
import os
import glob

# Define the allowed values for each parameter
# files = ['model_name=gpt-4o-mini-use_system=False-use_sample=True-use_response=False-number_of_shots=3-use_temperature=False-use_restriction=False.json', 
        #  'model_name=gpt-4o-mini-use_system=True-use_sample=True-use_response=False-number_of_shots=5-use_temperature=False-use_restriction=True.json', 
# files = ['model_name=gpt-4o-use_system=False-use_sample=True-use_response=True-number_of_shots=3-use_temperature=False-use_restriction=True.json', 
        #  'model_name=gpt-4o-use_system=True-use_sample=False-use_response=False-number_of_shots=5-use_temperature=False-use_restriction=False.json', 
        #  'model_name=mixtral-use_system=False-use_sample=True-use_response=False-number_of_shots=5-use_temperature=False-use_restriction=True.json', 
files = ['model_name=mixtral-use_system=True-use_sample=True-use_response=True-number_of_shots=5-use_temperature=False-use_restriction=False.json',
# files = ['model_name=qwen-use_system=False-use_sample=False-use_response=False-number_of_shots=5-use_temperature=False-use_restriction=False.json', 
#          'model_name=qwen-use_system=False-use_sample=True-use_response=False-number_of_shots=5-use_temperature=False-use_restriction=True.json',
         'mixtral_ground_truth.json',
        #  'gpt-4o-mini_ground_truth.json',
        #  'qwen_ground_truth.json',
         'mixtral_ground_truth',]
        #  'qwen_ground_truth',]
        #  "gpt-4o-mini_ground_truth"
        #  'gpt-4o_ground_truth.json',
        #  "gpt-4o_ground_truth"]

dataset_names = ["SummEval", "SumPubMed", "HelpSteer2"]
# dataset_names = ["USR"]
results_dir = os.path.join("results", "definition")

for json_file in files:
        if "llama" in json_file:
            api = "llama"
        elif "gpt-4o-mini" in json_file:
            api = "gpt-4o-mini"
        elif "gpt-4o" in json_file:
            api = "gpt-4o"
        elif "qwen" in json_file:
            api = "qwen"
        elif "mixtral" in json_file:
            api = "mixtral"
        else: 
            raise ValueError(f"API not found in {json_file}")
        if "use_system=True" in json_file:
            use_system = True
        else:
            use_system = False
        if ".json" not in json_file:
            json_file = "ground_truth"
        for dataset in dataset_names:
            data = dataset
            subprocess_list = ["poetry", "run", "python", os.path.join(".", "scripts", "scoring.py"), "--data", dataset, "--api", api, "--setting_name", json_file]
            if use_system:
                subprocess_list.append("--use_system")
            print(subprocess_list)
            print("==============================================================================================")
            subprocess.run(subprocess_list)
