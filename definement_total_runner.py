import subprocess
import os

# Define the allowed values for each parameter
api_functions = ["gpt-4o","gpt-4o-mini","llama","vicuna","mixtral","qwen"]
dataset_names = ["USR", "SummEval", "SumPubMed", "MT-Bench", "CommonGen"]
boolean_options = {
    "use_system": [True, False],
    "use_temperature": [True, False],
    "use_restriction": [True, False],
    "use_sample": [0, 1, 2],
}
number_of_shots_range = range(1, 6)

apis = ["mixtral","llama","gpt-4o","gpt-4o-mini"]
dataset = "USR"
for api in apis:
    for use_system in boolean_options["use_system"]:
        for use_temperature in [False]:
            for use_restriction in boolean_options["use_restriction"]:
                for use_sample in boolean_options["use_sample"]:
                    subprocess_list = ["poetry", "run", "python", os.path.join(".","scripts","definement.py"), "--data", dataset, "--api", api]
                    subprocess_list.extend([f"--{name}"for name,option in [("use_system", use_system), ("use_temperature", use_temperature), ("use_restriction", use_restriction)] if option])
                    if use_sample == 1:
                        subprocess_list.append("--use_sample")
                    elif use_sample ==2:
                        subprocess_list.extend(["--use_sample", "--use_response"])      
                    print(subprocess_list)
                    print("==============================================================================================")         
                    subprocess.run(subprocess_list)
