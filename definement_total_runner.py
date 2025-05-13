import subprocess
import os

# Define the allowed values for each parameter
api_functions = ["gpt-4o","gpt-4o-mini","llama","mixtral","qwen"]
dataset_names = ["USR", "SummEval", "SumPubMed", "HelpSteer2"]
boolean_options = {
    "use_system": [True, False],
    "use_temperature": [True, False],
    "use_restriction": [True, False],
    "use_sample": [0, 1, 2],
}
number_of_shots_range = range(1, 6)

# apis = ["qwen"]
dataset = "USR"
for api in api_functions:
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



# import subprocess
# import os

# dataset_names = ["USR", "SummEval", "SumPubMed", "HelpSteer2"]

# # Define the best settings for each model
# settings = [
#     {
#         "model_name": "qwen",
#         "use_system": False,
#         "use_sample": True,
#         "use_response": False,
#         "number_of_shots": 5,
#         "use_temperature": False,
#         "use_restriction": True,
#     },
#     {
#         "model_name": "qwen",
#         "use_system": False,
#         "use_sample": False,
#         "use_response": False,
#         "number_of_shots": 5,
#         "use_temperature": False,
#         "use_restriction": False,
#     },
#     {
#         "model_name": "mixtral",
#         "use_system": False,
#         "use_sample": True,
#         "use_response": False,
#         "number_of_shots": 5,
#         "use_temperature": False,
#         "use_restriction": True,
#     },
#     {
#         "model_name": "mixtral",
#         "use_system": True,
#         "use_sample": True,
#         "use_response": True,
#         "number_of_shots": 5,
#         "use_temperature": False,
#         "use_restriction": False,
#     },
#     {
#         "model_name": "gpt-4o",
#         "use_system": False,
#         "use_sample": True,
#         "use_response": True,
#         "number_of_shots": 5,
#         "use_temperature": False,
#         "use_restriction": True,
#     },
#     {
#         "model_name": "gpt-4o",
#         "use_system": True,
#         "use_sample": False,
#         "use_response": False,
#         "number_of_shots": 5,
#         "use_temperature": False,
#         "use_restriction": False,
#     },
#     {
#         "model_name": "gpt-4o-mini",
#         "use_system": False,
#         "use_sample": True,
#         "use_response": False,
#         "number_of_shots": 5,
#         "use_temperature": False,
#         "use_restriction": False,
#     },
#     {
#         "model_name": "gpt-4o-mini",
#         "use_system": True,
#         "use_sample": True,
#         "use_response": False,
#         "number_of_shots": 5,
#         "use_temperature": False,
#         "use_restriction": True,
#     },
#     {
#         "model_name": "llama",
#         "use_system": True,
#         "use_sample": True,
#         "use_response": False,
#         "number_of_shots": 5,
#         "use_temperature": False,
#         "use_restriction": False,
#     },
#     {
#         "model_name": "llama",
#         "use_system": False,
#         "use_sample": False,
#         "use_response": False,
#         "number_of_shots": 5,
#         "use_temperature": False,
#         "use_restriction": True,
#     },
# ]

# for dataset in dataset_names[2:3]:
#     for setting in settings:
#         # if setting["model_name"] == "llama" and dataset == "HelpSteer2":
            
#             subprocess_list = [
#                 "poetry", "run", "python", os.path.join(".", "scripts", "definement.py"),
#                 "--data", dataset,
#                 "--api", setting["model_name"],
#                 "--number_of_shots", str(setting["number_of_shots"]),
#             ]

#             # Add flags based on the settings
#             if setting["use_system"]:
#                 subprocess_list.append("--use_system")
#             if setting["use_temperature"]:
#                 subprocess_list.append("--use_temperature")
#             if setting["use_restriction"]:
#                 subprocess_list.append("--use_restriction")

#             # Handle use_sample and use_response flags
#             if setting["use_sample"]:
#                 subprocess_list.append("--use_sample")
#                 if setting["use_response"]:
#                     subprocess_list.append("--use_response")

#             # Print and run the command
#             print("Executing:", " ".join(subprocess_list))
#             print("=" * 90)
#             subprocess.run(subprocess_list)
