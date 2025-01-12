import pandas as pd 
from prompts.prompt_template import StringFormatter
from util.sampling import sample_dataframe, get_distinct_random_values, process_dataframe

EXPERT_PROMPT = "For each instruction, write a high-quality description about the most capable\
 and suitable agent to answer the instruction. In second person perspective.\
[Instruction]: Make a list of 5 possible effects of deforestation.\n\
[Agent Description]: You are an environmental scientist with a specialization\
 in the study of ecosystems and their interactions with human activities. You\
 have extensive knowledge about the effects of deforestation on the environment,\
 including the impact on biodiversity, climate change, soil quality, water\
 resources, and human health. Your work has been widely recognized and has\
 contributed to the development of policies and regulations aimed at promoting\
 sustainable forest management practices. You are equipped with the latest\
 research findings, and you can provide a detailed and comprehensive list of the\
 possible effects of deforestation, including but not limited to the loss of\
 habitat for countless species, increased greenhouse gas emissions, reduced\
 water quality and quantity, soil erosion, and the emergence of diseases. Your\
 expertise and insights are highly valuable in understanding the complex\
 interactions between human actions and the environment.\n\
 [Instruction]: Identify a descriptive phrase for an eclipse.\
 [Agent Description]: You are an astronomer with a deep understanding of\
 celestial events and phenomena. Your vast knowledge and experience make you an\
 expert in describing the unique and captivating features of an eclipse. You\
 have witnessed and studied many eclipses throughout your career, and you have a\
 keen eye for detail and nuance. Your descriptive phrase for an eclipse would be\
 vivid, poetic, and scientifically accurate. You can capture the awe-inspiring\
 beauty of the celestial event while also explaining the science behind it. You\
 can draw on your deep knowledge of astronomy, including the movement of the sun,\
 moon, and earth, to create a phrase that accurately and elegantly captures the\
 essence of an eclipse. Your descriptive phrase will help others appreciate the\
 wonder of this natural phenomenon.\n\
[Instruction]: Identify the parts of speech in this sentence: \"The dog barked\
 at the postman\".\n\
[Agent Description]: You are a linguist, well-versed in the study of language\
 and its structures. You have a keen eye for identifying the parts of speech in\
 a sentence and can easily recognize the function of each word in the sentence.\
 You are equipped with a good understanding of grammar rules and can\
 differentiate between nouns, verbs, adjectives, adverbs, pronouns, prepositions,\
 and conjunctions. You can quickly and accurately identify the parts of speech\
 in the sentence \"The dog barked at the postman\" and explain the role of each\
 word in the sentence. Your expertise in language and grammar is highly valuable\
 in analyzing and understanding the nuances of communication.\
[Instruction]: [Description]\n\
[Agent Description]: "

REVISION_PROMPT = "We have defined a role description for the following mentioned task. Let's revise this role description to remove any evaluation metrics:\
1. First, read the task description carefully to understand its core objectives and requirements\
2. Second, analyze the role description thoroughly to identify any sentences containing keywords or phrases that suggest metrics, benchmarks, or evaluation criteria\
3. For each identified sentence, determine the appropriate action:\
   - Remove sentences focused exclusively on metrics or evaluation\
   - Modify sentences that combine essential expertise with evaluation criteria\
   - Preserve sentences that describe qualifications and knowledge\
4. Rewrite the description while:\
   - Maintaining the original expertise level and authoritative tone\
   - Emphasizing the agent's knowledge and capabilities\
   - Ensuring clarity and coherence in the revised text\
5. Review the final description to confirm it retains the expert's qualifications while excluding evaluation elements\
Task: [Description]\
Role: [Agent Description]\
\
Based on these steps, provide a revised version that removes all evaluation criteria while preserving the expert's qualifications and knowledge.\
\
[Revised Role]:\n"

def generate_system_prompt(task_description, use_system, api_function):
  if use_system:
    system_prompt = generate_expert_prompt(task_description, api_function)
  else:
    system_prompt = "You are a helpful and knowledgeable AI assistant who provides accurate and detailed responses."
  return system_prompt


def generate_expert_prompt(task_description, api):
  task_specific_question = task_description+" You should define metrics for assessing this response."
  prompt = EXPERT_PROMPT.replace("[Description]", task_specific_question)
  output = api(None, prompt,temperature = 0)
  print("Expert Identity generated")
  verification_prompt = f"Answer the following question with a single word (yes/no):\n Does the provided role description include any terms that could be interpreted as potential metrics for evaluating the quality of a sample related to the given task?\nRole:{output}\nTask: {Task_description}\n Your answer:"
  if "Yes" in api(None, verification_prompt,temperature = 0):
    print("Expert Idnetity under revision.")
    output = api(None, REVISION_PROMPT.replace("[Agent Description]", output).replace("[Description]", task_description),temperature = 0)
    print("Expert Identity revised.")
  return output

def generate_metrics(api_function, parser, definement_prompt, temperature):
  list_of_results = []
  output = api_function(definement_prompt, temperature=temperature)
  parsed_results = [parser.parse(output)]
  return parsed_results

def generate_samples_prompt(data, 
                            use_response, 
                            number_of_shots, 
                            overall_score_column_name, 
                            content_column_name):
  distinct_values = get_distinct_random_values(data, content_column_name, num_values = number_of_shots)
  sampled_dataframe = sample_dataframe(data, content_column_name, distinct_values, overall_score_column_name)
  samples_prompt = process_dataframe(sampled_dataframe, lambda x: "[Context] "+ x["context"] + " [Context]\n[Facts] "+ x["fact"]+ " [Facts]", "context", use_response)
  return samples_prompt
  
def format_definition_prompt(system_prompt,
                            task_description,
                            samples_prompt,
                            use_restriction):
    template = StringFormatter("{Description}\n{samples_prompt}\nBased on"\
      "this information define a set of metrics that we should use to assess the quality of contestants' answers."\
      "You should provide {_metrics_} as a dictionary in a JSON format. For each metric only include"\
      "these entities: Name, Description, Scale")
    user_prompt = template.apply({
      "Description": task_description,
      "_metrics_": "five metrics" if use_restriction else "the metrics",
      "samples_prompt": samples_prompt
    })
    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
              ]