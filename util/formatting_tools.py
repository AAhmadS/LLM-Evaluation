import json
import os
import pandas as pd
from typing import List
from models.metric import Metric
from prompts.prompt_template import (
MetricsList, 
define_metrics,
RoleDescription,
role_description_output
)

#################Definement api parsers#######################
def gpt_api_parser(metrics: MetricsList):
  metrics_list = []
  for abs_metric in metrics.metrics:
    name = abs_metric.name
    description = abs_metric.description
    scale = abs_metric.scale
    metric = Metric(name, description, scale)
    metrics_list.append(metric)
  return metrics_list

def vicuna_spot_list(api_response: str):
  return api_response[api_response.find('['):api_response.rfind(']')+1]
    
def vicuna_api_parser(api_response: str):
  metrics_list = []
  structured_response = json.loads(vicuna_spot_list(api_response))
  for metric_dict in structured_response:
    metric = Metric(**metric_dict['metric'])
    metrics_list.append(metric)
  return metrics_list 

def extract_arguments(api_response: str):
  clean_response = api_response[api_response.find('<tool_call>')+1:api_response.rfind('</tool_call>')]
  response_dict = json.loads(clean_response)
  arguments = response_dict['arguments']
  return arguments

def gen_api_parser(api_response: str):
  list_of_string_metrics = extract_arguments(api_response)
  metrics_list  = define_metrics(list_of_string_metrics)
  return metrics_list 

def llama_api_parser(api_response: dict):
  metrics_list = []
  metrics = json.loads(api_response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["metrics"]
  if type(metrics) == str:
    metrics = json.loads(metrics)
  for metric_dict in metrics:    
    metric = Metric(**metric_dict["metric"])
    metrics_list.append(metric)
  return metrics_list

##################Idnetity api parsers########################
def gpt_identity_parser(api_response: RoleDescription):
  return api_response.role_description

def vicuna_identity_parser(api_response: str):
  structured_response = json.loads(api_response)
  role_description = structured_response["expert identity"]
  return role_description

def llama_identity_parser(api_response: dict):
  try:
    role_description = json.loads(api_response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["role_description"]
    return role_description
  except:
      role_description = api_response["choices"][0]["message"]["content"]
      return role_description

def gen_identity_parser(api_response: str):
  role_description = extract_arguments(api_response)
  role_description = role_description_output(role_description)
  return role_description
###################Parser classes#############################
class DefinitionParser:
  
  def __init__(self, api_name):
    self.parser =  self.get_parser(api_name)
    
  def get_parser(self, api_name):
    if api_name == 'vicuna':
      return vicuna_api_parser
    elif api_name == 'mixtral':
      return vicuna_api_parser
    elif 'gpt' in api_name:
      return gpt_api_parser
    elif 'llama' in api_name:
      return llama_api_parser
    else:
      return gen_api_parser  

class IdentityParser:
  
  def __init__(self, api_name):
    self.parser =  self.get_parser(api_name)
    
  def get_parser(self, api_name):
    if api_name == 'vicuna':
      return vicuna_identity_parser
    elif 'gpt' in api_name:
      return gpt_identity_parser
    elif 'llama' in api_name:
      return llama_identity_parser
    elif api_name == 'mixtral':
      return vicuna_identity_parser
    else:
      return gen_identity_parser  

def extract_json_from_content(content):
  """Extracts JSON data from a text file."""
  try:
      # Assuming the JSON data is enclosed in curly braces {}
      start_index = content.find('```json')
      end_index = content.rfind('```')
      if start_index != -1 and end_index != -1:
        json_string = content[start_index+8 : end_index]
        return json.loads(json_string)
      else:
        print(f"No valid JSON found in {content}")
        return content

  except json.JSONDecodeError as e:
    print(f"Error decoding JSON in {content}: {e}")
    return content

def extract_json_from_list_of_contents(list_of_contents):
  return [extract_json_from_content(content) for content in list_of_contents]

def create_dataframe(criteria_data):
    # Initialize an empty list to store rows
    rows = []

    # Handle the case where data is a dictionary of dictionaries
    if isinstance(criteria_data, dict):
        # Check if it contains a single key with a list of dictionaries (e.g., {'Metrics': [...]})
        if len(criteria_data) == 1 and isinstance(next(iter(criteria_data.values())), list):
            # Extract the list of dictionaries
            items = next(iter(criteria_data.values()))
            for item in items:
                rows.append({
                    'name': item.get('Name', ''),
                    'description': item.get('Description', ''),
                    'scale': item.get('Scale', '')
                })
        # Handle dictionary of dictionaries directly (first example format)
        else:
            for item in criteria_data.values():
                rows.append({
                    'name': item.get('Name', ''),
                    'description': item.get('Description', ''),
                    'scale': item.get('Scale', '')
                })

    # Handle the case where data is a list of dictionaries
    elif isinstance(criteria_data, list):
        for item in criteria_data:
            rows.append({
                'name': item.get('Name', ''),
                'description': item.get('Description', ''),
                'scale': item.get('Scale', '')
            })
    return rows  # Return rows instead of a DataFrame

def parse_metrics(metrics_dataframe):
  metrics_dataframe['metrics'] = metrics_dataframe["results"].apply(extract_json_from_list_of_contents)
  rows = []
  for content in metrics_dataframe["metrics"]:
    for metrics in content:
      rows += create_dataframe(metrics)

  metric_description_df = pd.DataFrame(rows, columns=['name', 'description', 'scale'])
  return metric_description_df