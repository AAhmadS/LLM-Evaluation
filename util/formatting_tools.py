import json
import os
import pandas as pd
from typing import List
from models.metric import Metric
import re
from prompts.prompt_template import (
MetricsList, 
define_metrics,
RoleDescription,
role_description_output,
MetricScore,
MetricInstruction,
set_metric_instruction,
get_metric_score,
MetricTags,
AllMetricsTags
)

#################Local tools##################################
def spot_list(api_response: str):
  return api_response[api_response.find('['):api_response.rfind(']')+1]

def extract_arguments(api_response: str):
  clean_response = api_response[api_response.find('<tool_call>')+1:api_response.rfind('</tool_call>')]
  response_dict = json.loads(clean_response)
  arguments = response_dict['arguments']
  return arguments

def modify_qwen_string(input_string):
    """
    Finds the second "]" from the right, checks if two characters before it is "}". 
    If false, replaces the "}" before the "]" with "}}".
    """
    try:
        # Find the second ']' from the right
        bracket_index = input_string.rfind(']')
        
        # Check if the second ']' is found
        if bracket_index == -1:
            return input_string  # Return original string if not found

        modified_string = input_string[:bracket_index] + "}" + input_string[bracket_index:]
        return modified_string
    except IndexError:
        return input_string
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

def vicuna_api_parser(api_response: str):
  metrics_list = []
  structured_response = json.loads(spot_list(api_response))
  for metric_dict in structured_response:
    metric = Metric(**metric_dict['metric'])
    metrics_list.append(metric)
  return metrics_list 

def gen_api_parser(api_response: str):
  list_of_string_metrics = extract_arguments(api_response)
  metrics_list  = define_metrics(list_of_string_metrics)
  return metrics_list 

def llama_api_parser(api_response):
  api_response = api_response.choices[0].message.tool_calls[0].function.arguments
  metrics_list = []
  metrics = json.loads(api_response)["metrics"]
  if type(metrics) == str:
    metrics = json.loads(metrics)
  for metric_dict in metrics:
    metric = Metric(**metric_dict["metric"])
    metrics_list.append(metric)
  return metrics_list

def qwen_api_parser(api_response):
  api_response = api_response.choices[0].message.tool_calls[0].function.arguments
  metrics_list = []
  metrics = json.loads(api_response)["metrics"]
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
  try:
    structured_response = json.loads(api_response)
    role_description = structured_response["expert identity"]
    return role_description
  except: 
    role_description = api_response
    return role_description

def llama_identity_parser(api_response):
  
  try:
    try:
      role_description = json.loads(api_response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["role_description"]
      return role_description
    except:
      try:
        role_description = json.loads(api_response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["expert identity"]
        return role_description      
      except:
        raise ValueError(f"format of response not recognized {api_response}")
  except:
      role_description = api_response["choices"][0]["message"]["content"]
      return role_description

def qwen_identity_parser(api_response):
  try:
    role_description = json.loads(api_response.choices[0].message.tool_calls[0].function.arguments)["role_description"]
    return role_description
  except:
      role_description = api_response.choices[0].message.content
      return role_description

def gen_identity_parser(api_response: str):
  role_description = extract_arguments(api_response)
  role_description = role_description_output(role_description)
  return role_description

###################Instruction Generation parser##############
def gpt_instruction_parser(api_response: MetricInstruction):
  instructions = api_response.steps
  steps = ""
  for step in instructions:
    steps+= "\n"+ str(step.step_id) + "- "+ step.description
  return steps

def vicuna_instruction_parser(api_response: str):
  structured_response = json.loads(api_response) # this should be a list of dictionaries
  steps = ""
  for step_dict in structured_response:
    step_dict = step_dict["step"]
    steps+= "\n"+str(step_dict["step_id"]) + "- "+ step_dict["description"]
  
  return steps
  
def llama_instruction_parser(api_response):
  try:
    function_call = api_response.choices[0].message.content
    function_call = function_call[function_call.index('{'):function_call.rindex('}')+1]
    if type(function_call) == str:      
      function_call = json.loads(function_call)
    steps = function_call["steps"]
    if type(steps) == str:
      steps = json.loads(steps)
    steps = set_metric_instruction(steps)
    return steps
  except:
      print("in exception")
      instructions = api_response.choices[0].message.content
      return instructions

def qwen_instruction_parser(api_response):
  try:
    function_call = api_response.choices[0].message.content[15:]
    if type(function_call) == str:
      if "required" in function_call:
        function_call = function_call[:function_call.rindex(',')] + function_call[function_call.rindex(']')+1:]      
      try:
        function_call = json.loads(function_call)
      except:
        function_call = json.loads(modify_qwen_string(function_call))
    arguments = function_call["arguments"]
    if type(arguments) == str:
      arguments = json.loads(arguments)
    steps = arguments["steps"]
    steps = set_metric_instruction(steps)
    return steps
  except:
      print("in exception")
      instructions = api_response.choices[0].message.content
      return instructions

def gen_instruction_parser(api_response: str):
  instructions = api_response
  return instructions

####################Scoring parser############################
def gpt_scoring_parser(api_response: MetricScore):
  return{
    "reasoning": api_response.reasoning,
    "score": api_response.score
  }

def vicuna_scoring_parser(api_response: str):
  structured_response = json.loads(api_response) # this should a dictionary of the result of an evaluation
  structured_response = structured_response["evaluation result"]
  return {
    "reasoning": structured_response["reasoning"],
    "score": structured_response["score"]
  }
  
def llama_scoring_parser(api_response):
  try:
    evaluation = json.loads(api_response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["evaluation"]
    evaluation_result = get_metric_score(evaluation)
    return evaluation_result
  except:
      response = api_response["choices"][0]["message"]["content"]
      try:
        evaluation = spot_list(response)
        return get_metric_score(evaluation)
      except:
        raise Exception("Could not parse response")

def qwen_scoring_parser(api_response):
  try:
    function_call = api_response.choices[0].message.content
    if "!function_call:" in function_call:
      if function_call.count("!function_call:") >= 2:
        function_call = function_call.split("!function_call:")[-1]
        function_call = function_call[:function_call.rindex('}')+1]
        if function_call[-2] != "}":
          function_call = function_call + "}"
      else:
        function_call = function_call.replace("!function_call:", "")
      function_call = json.loads(function_call)
      function_call = function_call["arguments"]
      if type(function_call) == str:
        function_call = json.loads(function_call)
      function_call = function_call["evaluation"]
    if type(function_call) == str:  
      if function_call[0] != "{":
        function_call = "{" + function_call
      if "evaluation_result" in function_call:
        function_call = function_call.replace("evaluation_result", "evaluation result")
    evaluation = get_metric_score(function_call)
    return evaluation
  except:
      print("in exception")
      instructions = api_response.choices[0].message.content
      integer_value = re.findall(r'\d+', instructions.split()[-1])[-1]
      return {
        "score": integer_value,
        "reasoning": instructions
      }

def gen_scoring_parser(api_response: str):
  evaluation = api_response
  return evaluation
     
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
    elif 'qwen' in api_name:
      return qwen_api_parser
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
    elif api_name == 'qwen':
      return qwen_identity_parser
    else:
      return gen_identity_parser  

class InstructionParser:
  def __init__(self, api_name):
    self.parser =  self.get_parser(api_name)
    
  def get_parser(self, api_name):
    if api_name == 'vicuna':
      return vicuna_instruction_parser
    elif 'gpt' in api_name:
      return gpt_instruction_parser
    elif 'llama' in api_name:
      return llama_instruction_parser
    elif api_name == 'mixtral':
      return vicuna_instruction_parser
    elif api_name == 'qwen':
      return qwen_instruction_parser
    else:
      return gen_instruction_parser
    
class ScoringParser:
  def __init__(self, api_name):
    self.parser =  self.get_parser(api_name)

  def get_parser(self, api_name):
    if api_name == 'vicuna':
      return vicuna_scoring_parser
    elif 'gpt' in api_name:
      return gpt_scoring_parser
    elif 'llama' in api_name:
      return llama_scoring_parser
    elif api_name == 'mixtral':
      return vicuna_scoring_parser
    elif api_name == 'qwen':
      return qwen_scoring_parser
    else:
      return gen_scoring_parser
    
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

##################Tag Parser#################################

def gpt_tag_parser(metricstags: AllMetricsTags):
  metric_tags = []
  for metric_tag in metricstags.metrics_tags:
    name = metric_tag.name
    tags = metric_tag.tags
    metric_tags.append({"name": name, "tags": tags})
  return metric_tags