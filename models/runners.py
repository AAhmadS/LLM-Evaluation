import random
import numpy as np
import pandas as pd
from util.prompting_utils import (
  generate_system_prompt, 
  generate_samples_prompt, 
  format_definition_prompt,
  # format_scoring_prompt,
  generate_metrics
)
from util.formatting_tools import parse_metrics
from util.similarity_tools import merge_similar_metrics
from results.save_and_load import save_results

def end_to_end_definement(api_function, 
                          parser,
                          run_configuration,
                          dataset, 
                          seed = 42):
  random.seed(seed)
  use_system, use_sample, use_response, number_of_shots, use_temperature, use_restriction = run_configuration.dump()
  
  # extract dataset information
  data = dataset.data
  dataset_name = dataset.dataset_name
  task_description = dataset.task_description
  overall_score_column_name = dataset.overall_score_column_name
  content_column_name = dataset.content_column_name
  
  # settings_string for saving in the end
  saving_settings_string = run_configuration.to_string()

  # check for expert identity
  system_prompt = generate_system_prompt(task_description, use_system,api_function)
  print("system_prompt:\n", system_prompt)
  
  # check for using samples
  if use_sample:
    samples_prompt = generate_samples_prompt(data,
                                             use_response, 
                                             number_of_shots, 
                                             overall_score_column_name, 
                                             content_column_name
                                             )

  else:
    samples_prompt = ""
  print("samples_prompt:\n", samples_prompt)
  
  # finalizing the definition prompt
  definition_chain_of_prompts = format_definition_prompt(system_prompt,
                                                         task_description,
                                                         samples_prompt,
                                                         use_restriction
                                                         )
  print("definition_chain_of_prompts:\n", definition_chain_of_prompts)
  
  # defining metrics
  metrics = []
  metrics_dataframe = pd.DataFrame(columns=['metrics'])
  
  # check for using temperature (when generating the metrics with different temperatures)
  if use_temperature:
    # list of temperatures to use
    temperatures = np.arange(0.5, 0.9, 0.1, dtype=float)
  else:
    temperatures = [0.7] * 5

  for temperature in temperatures:
    metrics.extend(generate_metrics(api_function, parser, definition_chain_of_prompts, temperature))
    print(f"---Generated Metrics with Temperature {temperature}---")

  new_row = {'metrics': metrics}
  metrics_dataframe = pd.concat([metrics_dataframe, pd.DataFrame([new_row])], ignore_index=True)
  
  print("---Saving Metrics---")
  save_results("definition",metrics, dataset_name, saving_settings_string)
  
  return metrics
  
  
def similarity_based_filtering(metrics, 
                               use_similarity, 
                               similarity_threshold, 
                               meaningfulness_threshold
                               ):

  # parse the metrics
  print('---Metrics Parsed---')
  # check for using similarity
  metrics = merge_similar_metrics(metrics, use_similarity, similarity_threshold, meaningfulness_threshold)
  print(f"---Finalized Metrics: {metrics['name']}---")
  return metrics

# def instruction_generation(api_function,
#                            metrics_list,
#                            seed=42):
#   random.seed(seed)
#   # define instruction for metrics
  
  
# def end_to_end_evaluation(api_function,
#                           dataset,
#                           metrics_set,
#                           run_configuration,
#                           seed=42
#                           ):
  
#   random.seed(seed)
#   # extract dataset information
#   data = dataset.selected_subset
#   dataset_name = dataset.dataset_name
#   task_description = dataset.task_description
#   overall_score_column_name = dataset.overall_score_column_name
#   content_column_name = dataset.content_column_name
    
#   # settings_string for saving in the end
#   saving_settings_string = run_configuration.to_string()

#   # check for expert identity
#   system_prompt = generate_system_prompt(task_description, True, api_function)
#   print("system_prompt:\n", system_prompt)
  
#   scoring_chain_of_prompts = format_scoring_prompt(system_prompt,
#                                                       task_description,
#                                                       samples_prompt
#                                                       )
#   print("definition_chain_of_prompts:\n", definition_chain_of_prompts)
#   metrics = []
#   metrics_dataframe = pd.DataFrame(columns=['results'])
  
#   # five times generation, hence two times appearance seems sufficiently meaningful.(we might not use this in final version)
#   meaningfulness_threshold = 2

#   # check for using temperature (when generating the metrics with different temperatures)
#   if use_temperature:
#     # list of temperatures to use
#     temperatures = np.arange(0.5, 0.9, 0.1, dtype=float)
#   else:
#     temperatures = [0.7] * 5

#   for temperature in temperatures:
#     metrics.extend(generate_metrics(api_function, definition_chain_of_prompts, temperature))
#     print(f"---Generated Metrics with Temperature {temperature}---")

#   new_row = {'results': metrics}
#   metrics_dataframe = pd.concat([metrics_dataframe, pd.DataFrame([new_row])], ignore_index=True)
#   print("---Saving Metrics---")
#   save_results("definition",metrics, dataset_name, saving_settings_string)