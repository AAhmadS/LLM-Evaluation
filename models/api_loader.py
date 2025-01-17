from openai import OpenAI
from pydantic import BaseModel
import os

###############################################################################
def gpt_api(messages, response_format, temperature=0.7):
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
    )
    client.beta.chat.completions.parse

    chat_completion = client.beta.chat.completions.parse(
        messages=messages,
        model=model,
        temperature=temperature,
        response_format=response_format
    )
    return chat_completion.choices[0].message.parsed
###############################################################################
from llamaapi import LlamaAPI
import json

def llama_api(messages, response_format, temperature=0.7):
    llama = LlamaAPI(os.getenv("LLAMA_API_KEY"))
    
    # Build the API request
    api_request_json = {
        "model": "llama3.1-70b",
        "messages": messages,
        "tools" : [response_format],
        "temperature" : temperature,
        "max_tokens" : 4096,
        "stream": False,
        "function_call": response_format["function"]["name"],
    }
    # Execute the Request
    response = llama.run(api_request_json)

    return response.json()
###############################################################################
import os
from mistralai import Mistral

def mistral_api(messages, response_format, temperature=0.7):
    api_key = os.getenv("MISTRALAI_API_KEY")
    model = "open-mixtral-8x22b"
    client = Mistral(api_key=api_key)

    messages[-1]['content'] += response_format()

    chat_response = client.chat.complete(
        model = model,
        messages = messages,
        temperature = 0.7,
        max_tokens = 4096,
        stream = False,
    )

    return chat_response.choices[0].message.content
###############################################################################
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torch
from functools import partial

## Vicuna-13B-v1.5 16k
def vicuna_chat_completion(messages, model, formatting_prompt, tokenizer, temperature=0.7, top_p=1):
    conversation = ""
    for message in messages:
        if message['role'] == 'system':
            conversation += f"SYSTEM: {message['content']}\n"
        elif message['role'] == 'user':
            conversation += f"USER: {message['content']}\n"
        else:
            conversation += f"ASSISTANT: {message['content']}\n"
    conversation += formatting_prompt
    conversation += "ASSISTANT: "

    inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1000, temperature=temperature, top_p = top_p, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("ASSISTANT:")[-1].strip()

def chat_with_accelerate(messages, model, tokenizer, tools, max_new_tokens=4096, temperature=0.7, top_p=1):

    # Prepare input
    inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tools=tools,
                return_dict=True,
                return_tensors="pt",
    ).to("cuda")
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

    # Decode and return response
    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response

###############################################################################
def load_api(api_name):
    if 'gpt' in api_name:
        os.environ["OPENAI_MODEL"] = api_name
        return gpt_api
    elif api_name == 'vicuna':
        model_name = "lmsys/vicuna-13b-v1.5-16k"
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return partial(vicuna_chat_completion, model=model, tokenizer=tokenizer)
    elif api_name == 'llama':
        return llama_api
    elif api_name == 'qwen':
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-32B-Instruct", device_map="auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
        return partial(chat_with_accelerate, model=model, tokenizer=tokenizer)
    elif api_name == 'mixtral':
        return mistral_api
    else:
        raise ValueError(f"Unknown API name: {api_name}")
