import argparse
import json
import os
import random
import sys

import numpy as np
import openai
import pandas as pd
import torch
from dotenv import load_dotenv
from procoder.functional import (
    add_refnames,
    format_multiple_prompts,
)
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append('../data_construction')
from toolemu.prompts.agent import *
from toolemu.tools import get_toolkits_by_names

sys.path.append('../helper')
from utils import openai_chat_completion_with_retry, print_api_usage, VLLM, claude_chat_completion_with_retry


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True,
                        help='Path of the data in json format.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save the probing results in csv format.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Start index of the cases to evaluate.')
    parser.add_argument('--num', type=int, default=1,
                        help='Number of cases to evaluate. If -1, evaluate all remaining cases.')
    parser.add_argument('--specific-case-name', type=str, default=None,
                        help='If not None, only evaluate the case with the given name.')
    parser.add_argument('--prompt-type', type=str, choices=['naive', 'privacy_enhanced'],
                        help='The type of the prompt to use for the agent.')
    parser.add_argument('--model', type=str, required=True,
                        choices=['gpt-4-1106',
                                 'gpt-3.5-turbo-1106',
                                 'claude-3-haiku-20240307',
                                 'claude-3-sonnet-20240229',
                                 'mistralai/Mistral-7B-Instruct-v0.2',
                                 'HuggingFaceH4/zephyr-7b-beta',
                                 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                                 'meta-llama/Meta-Llama-3-8B-Instruct',
                                 'meta-llama/Meta-Llama-3-70B-Instruct'])
    parser.add_argument('--gpu-num', type=int, default=1,
                        help='Number of GPUs to use for VLLM.')
    parser.add_argument('--hf-cache-dir', type=str,
                        help='The cache directory for the Hugging Face model.')

    return parser.parse_args()


def prepare_agent_prompt(
        prompt_type,
        user_name,
        user_email,
        user_instruction,
        toolkits,
        executable_trajectory,
        final_action
):
    toolkits = get_toolkits_by_names(toolkits)
    all_tools = []
    for toolkit in toolkits:
        all_tools += toolkit.tools

    toolkit_strings = "\n".join(
        [toolkit.create_description("medium") for toolkit in toolkits]
    )
    tool_names = ", ".join([tool.name for tool in all_tools])

    inputs = dict(toolkit_descriptions=toolkit_strings, tool_names=tool_names)

    add_refnames(AGENT_DUMMY_VARS, inputs, include_brackets=False)

    if prompt_type == 'naive':
        system_info = AGENT_NAIVE_SYSTEM_INFO
        prompt_instruction = AGENT_NAIVE_PROMPT
    elif prompt_type == 'privacy_enhanced':
        system_info = AGENT_PRIVACY_ENHANCED_SYSTEM_INFO
        prompt_instruction = AGENT_PRIVACY_ENHANCED_PROMPT
    else:
        raise ValueError('[Error] Unknown prompt type: {prompt_type}.')

    system_info, prompt_instruction = format_multiple_prompts(
        [system_info, prompt_instruction], inputs, include_brackets=[False, True]
    )

    prompt_instruction = prompt_instruction.format(
        user_name=user_name,
        user_email=user_email,
        input=user_instruction,
        agent_scratchpad=executable_trajectory
    )

    final_prompt = system_info + '\n\n' + prompt_instruction

    final_prompt = final_prompt[:final_prompt.find('## Start the Execution')]
    final_prompt += ('## Start the Execution\n'
                     f'User Input: {user_instruction}\nPast Trajectory:\n{executable_trajectory}\n\n'
                     f'Please output your **NEXT** [Action]/[Action Input] following the provided [Scratchpad], '
                     f'directly start your response with your [Thought] for the current iteration. '
                     f'The [Action] should be {final_action}.')

    return final_prompt


def post_process(s):
    if s.startswith('<|start_header_id|>assistant<|end_header_id|>'):
        # Remove the assistant header for Llama-3.
        s = s[len('<|start_header_id|>assistant<|end_header_id|>'):].strip()
        s = s[:s.find('<|eot_id|>')]
    # Only keep the Action and Action Input.
    if 'Observation:' in s:
        s = s[:s.find('Observation:')]
    if 'Final Answer:' in s:
        s = s[:s.find('Final Answer:')]
    if '}' in s:
        s = s[:s.find('}') + 1]
    return s


@print_api_usage
def main():
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = prepare_args()
    load_dotenv()

    if 'gpt' in args.model:
        openai.api_key = os.environ['OPENAI_API_KEY']
        openai.api_base = os.environ['OPENAI_API_BASE']
        openai.api_type = os.environ['OPENAI_API_TYPE']
        openai.api_version = os.environ['OPENAI_API_VERSION']

    if 'mistral' in args.model or 'llama' in args.model or 'zephyr' in args.model or 'vicuna' in args.model:
        vllm_engine = VLLM(
            model=args.model,
            tensor_parallel_size=args.gpu_num,
            trust_remote_code=True,
            max_num_batched_tokens=16000,
            max_new_tokens=1000,
            temperature=0,
            download_dir=args.hf_cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.hf_cache_dir)

    with open(args.input_path, 'r') as f:
        data = json.load(f)

    if args.specific_case_name:
        for i, case in enumerate(data):
            if case['name'] == args.specific_case_name:
                args.start_index = i
                end_index = i + 1
                break
        else:
            raise ValueError(f'Error: The specific case name {args.specific_case_name} is not found.')
    else:
        if args.num == -1:
            end_index = len(data)
        else:
            end_index = min(args.start_index + args.num, len(data))

    result = {
        'name': [],
        'final_action': [],
    }

    for i in tqdm(range(args.start_index, end_index)):
        result['name'].append(data[i]['name'])
        agent_prompt = prepare_agent_prompt(
            prompt_type=args.prompt_type,
            user_name=data[i]['trajectory']['user_name'],
            user_email=data[i]['trajectory']['user_email'],
            user_instruction=data[i]['trajectory']['user_instruction'],
            toolkits=data[i]['trajectory']['toolkits'],
            executable_trajectory=data[i]['trajectory']['executable_trajectory'],
            final_action=data[i]['trajectory']['final_action']
        )

        if 'gpt' in args.model:
            final_action = openai_chat_completion_with_retry(
                model=args.model, messages=[{'role': 'user', 'content': agent_prompt}],
                max_tokens=400, temperature=0.0)
            final_action = final_action.choices[0].message['content'].strip()
        elif 'claude' in args.model:
            final_action = claude_chat_completion_with_retry(
                engine=args.model, messages=[{'role': 'user', 'content': agent_prompt}],
                max_tokens=400, temperature=0.0)
            final_action = final_action.content[0].text.strip()
        elif 'mistral' in args.model or 'llama' in args.model or 'zephyr' in args.model or 'vicuna' in args.model:
            inputs = [agent_prompt]
            inputs_in_chat_template = []
            if 'vicuna' in args.model:
                for input_text in inputs:
                    inputs_in_chat_template.append(f'User: {input_text}\n Assistant:')
            else:
                for input_text in inputs:
                    inputs_in_chat_template.append(
                        tokenizer.apply_chat_template([{'role': 'user', 'content': input_text}], tokenize=False)
                    )
            output = vllm_engine.generate(inputs_in_chat_template).generations
            final_action = output[0][0].text.strip()
        else:
            raise NotImplementedError

        # Post-process the model output.
        result['final_action'].append(post_process(final_action))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    try:
        pd.DataFrame(result).to_csv(args.output_path, index=False)
    except Exception as e:
        print(f'Error: {e}')
        with open(args.output_path.replace('.csv', 'json'), 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    main()
