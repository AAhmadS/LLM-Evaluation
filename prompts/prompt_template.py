from pydantic import BaseModel, Field
from typing import List

class StringFormatter:
    def __init__(self, template):
        """
        Initialize the StringFormatter with a template string.
        :param template: A string with placeholders enclosed in {}.
        """
        self.template = template

    def apply(self, values):
        """
        Replace placeholders in the template with corresponding values from the dictionary.
        :param values: A dictionary containing placeholder keys and their replacement values.
        :return: The formatted string with placeholders replaced.
        """
        try:
            return self.template.format(**values)
        except KeyError as e:
            raise ValueError(f"Missing value for placeholder: {e}") from None
        
def retrieve_set_of_tools(api_name):
    if api_name == "vicuna":
        return [get_vicuna_metric_prompt, get_vicuna_instruction_prompt, get_vicuna_score_prompt]
    elif "gpt" in api_name:
        return [MetricsList, MetricInstruction, MetricScore]
    else:
        return [define_metrics, set_metric_instruction, get_metric_score]

####################GPT-API Prompt Formatting Classes#####################
class simple_response(BaseModel):
    response: str = Field(description="your response to the prompt.")
    
class role_description(BaseModel):
    role_description: str = Field(description="description of the role.")
    
class AbstractMetric(BaseModel):
    name: str = Field(description="name of the metric.")
    description: str = Field(description="description of the metric. What does it measure?") 
    scale: str = Field(description="scale of the metric. What are the possible values or range?")
    
class MetricsList(BaseModel):
    metrics: List[AbstractMetric] = Field(description="List of metrics to be evaluated.")
    
class AbstractStep(BaseModel):
    step_id: str = Field(description="step number.")
    description: str = Field(description="description of the step. What should be done?")
    
class MetricInstruction(BaseModel):
    steps: List[str] = Field(description="List of steps to be evaluated.")
    
class MetricScore(BaseModel):
    reasoning: str = Field(description="reasoning behind the score.")
    score: float = Field(description="score based on the provided scale.")

####################Vicuna-API Prompt Formats##################

def get_vicuna_instruction_prompt():
    return'''Always use the following structure:

[
  {
    "step": {
      "step_id": "<step number>",
      "description": "<description of the step>"
    }
  }
]'''

def get_vicuna_metric_prompt():
    return'''Always use the following structure:
[
  {
    "metric": {
      "name": "<metric name>",
      "description": "<description of the metric>",
      "scale": "<scale of the metric>"
    }
  }
]'''

def get_vicuna_score_prompt():
    return'''Provide the reasoning and score using the following structure:

{
  "evaluation result": {
    "reasoning": "<reasoning behind the score>",
    "score": <score based on the provided scale>
  }
}'''


####################Other Prompt Formatting functions#######################
from models.metric import Metric
import json
def define_metrics(metrics: List[str]) -> List[Metric]:
    """
    Recieves a string of metrics in the following structure:
    [
        {
            "metric": {
                "name": "<metric name>",
                "description": "<description of the metric>",
                "scale": "<scale of the metric>"
            }
        }
    ]
    
    and returns a list of metrics.
    Args:
        metrics: A string of metrics in the above structure.
    Returns:
        A list of Metric objects        
    """
    metric_list = []
    for metric_string in metrics:
        metric_json = json.loads(metric_string)
        metric_list.append(Metric(**metric_json["metric"]))
    return metric_list  # A real function should probably actually get the temperature!

def set_metric_instruction(steps: List[str]) -> str:
    """
    Recieves a string of steps in the following structure:
    [
        {
            "step": {
                "step_id": "<step number>",
                "description": "<description of the step>"
            }
        }
    ]

    and returns the suitable format for the metric.
    Args:
        stesp: The structured ordered steps.
    Returns:
        A string of steps in the suitable format for the metric.
    """
    steps_str = ""
    for step in steps:
        step = json.loads(step)
        step_id = step["step"]["step_id"]
        description = step["step"]["description"]
        steps_str += f"{step_id}-{description}\n"
    return steps_str   # A real function should probably actually get the wind speed!

def get_metric_score(evaluation: str) -> int:
    """
    Recieves a string of the evaluation in the following structure:
    {
        "evaluation result": {
            "reasoning": "<reasoning behind the score>",
            "score": <score based on the provided scale>
        }
    }
    and returns the score.
    Args:
        evaluation: The structured evaluation.
    Returns:
        The score.
    """
    evaluation_dict = json.loads(evaluation)
    score = evaluation_dict["evaluation result"]["score"]
    return int(score)