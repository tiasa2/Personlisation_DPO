import random
import os
import argparse
import time
from datetime import datetime
from tqdm import tqdm
import torch

from transformers import AutoTokenizer
from openai import OpenAI
from vllm import LLM, SamplingParams



import re
def question_response(axioms):
    output = []
    pattern = r"Question:\s*(.+?),\s*Response:\s*(.+?)(?=\n|$)"
    pairs = re.findall(pattern, axioms)
    # Show output
    for i, (q, r) in enumerate(pairs, start=1):
        output.append(f"Question: {q}  Response: {r}\n")

    return output

available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
llm = LLM(
    model="Qwen/Qwen2.5-32B-Instruct",
    tensor_parallel_size=len(available_gpus) // 1,
    pipeline_parallel_size=1,
    trust_remote_code=True,
    max_model_len = 1024
    # enable_prefix_caching=True,
    # quantization='fp8',
    # kv_cache_dtype="fp8",
)

class LabelRestrictor:
    def __init__(self, tokenizer, label_texts):
        # assume each label is a single token (or use the last token)
        self.label_ids = [
            tokenizer.encode(" " + l, add_special_tokens=False)[-1]
            for l in label_texts
        ]

    def __call__(self, input_ids, scores: torch.Tensor):
        # scores: [vocab_size]
        mask = torch.full_like(scores, float("-inf"))
        mask[self.label_ids] = scores[self.label_ids]
        return mask


def classify_reasoning_against_axioms(reasoning_step, axioms):
    predicted_labels = []
    prompts = []
    for axiom in axioms: 
        prompt = f"""You are a logical reasoning validator. Given the axiom and reasoning step:
        1. Axiom : {axiom}
        2. Reasoning step {reasoning_step}

        Your task is to determine the logical relationship:

        - entailment: The reasoning step logically follows from the axiom (it must be true)
        - contradiction: The reasoning step contradicts the axiom (it cannot be true)
        - N/A: The reasoning step is not determined by the axiom (it might or might not be true)
        Just respond with one of the three labels: 'entailment', 'contradiction', or 'N/A'."""
        prompts.append(prompt)

    # responses = target_client.completions.create(
    #                 model=args.target_model_name_or_path.split("/")[-1],
    #                 prompt=prompts,
    #                 temperature=args.temperature,
    #                 top_p=args.top_p,
    #                 max_tokens=args.max_tokens_per_call,
    #                 stop=[args.step_word],
    #             ).choices

    labels = ['contradiction', 'entailment', 'N/A']
    
    target_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True)
    logits_processor = LabelRestrictor(target_tokenizer, labels)

    gen_output = llm.generate(prompts, SamplingParams(
                                    temperature=0, top_p=1,
                                    max_tokens=1,
                                    seed=0,
                                    n=1,
                                    logits_processors=[logits_processor]))
    # print('gen_output', gen_output[0].outputs[0].text)
    # predicted_labels.append(['contradiction', 'entailment', 'neutral'][logits.argmax(-1).item()])
    predicted_labels = [gen_output[index].outputs[0].text for index in range(len(prompts))]
    if 'contradiction' in predicted_labels:
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == 'contradiction':
                print(axioms[i])
        return axioms, predicted_labels, 0
    return axioms, predicted_labels, 1

def classify_relevant_reasoning_against_axioms(reasoning_step, axioms):
    predicted_labels = []
    prompts = []
    relevance_prompts = []
    for axiom in axioms:
        # relevance_prompt = f"""You are an axiom relevance classifier.

        #     Given:
        #     - Reasoning step: {reasoning_step}
        #     - Candidate axiom: {axiom}

        #     Task: Decide if this axiom is relevant to assessing the reasoning step.

        #     Guidelines:
        #     - Output "relevant" if the axiom directly constrains, supports, or contradicts the reasoning step, defines a term it uses, or provides a necessary condition the step depends on.
        #     - Output "irrelevant" if the axiom is off-topic, redundant background that doesn't change the truth of the step, or does not interact with entities/relations in the step.
        #     - If unsure, prefer "irrelevant".

        #     Output format: Return exactly one word: "relevant" or "irrelevant". No extra text.
        #     """
        relevance_prompt = f"""You are an axiom relevance classifier.

            Given:
            - Reasoning step: {reasoning_step}
            - Candidate axiom: {axiom}

            Task: Decide if this axiom is relevant to assessing the reasoning step.

            Guidelines:
            - Output "relevant" if the axiom related to the reasoning step, defines a term it uses, or provides a necessary condition the step depends on.
            - Output "irrelevant" if the axiom is unrelated to the reasoning step, or does not interact with entities/relations in the step.
            - If unsure, prefer "irrelevant".

            Output format: Return exactly one word: "relevant" or "irrelevant". No extra text.
            """

        relevance_prompts.append(relevance_prompt)
    
    relevance_labels = ['relevant', 'irrelevant']
    target_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True)
    logits_processor = LabelRestrictor(target_tokenizer, relevance_labels)

    relevance_gen_output = llm.generate(relevance_prompts, SamplingParams(
                                    temperature=0, top_p=1,
                                    max_tokens=1,
                                    seed=0,
                                    n=1,
                                    logits_processors=[logits_processor]))
    # print('gen_output', gen_output[0].outputs[0].text)
    # predicted_labels.append(['contradiction', 'entailment', 'neutral'][logits.argmax(-1).item()])
    relevance_predicted_labels = [relevance_gen_output[index].outputs[0].text for index in range(len(relevance_prompts))]

    relevant_axioms = [axioms[i] for i in range(len(axioms)) if relevance_predicted_labels[i] == ' relevant']
    if len(relevant_axioms) == 0:
        return relevant_axioms, relevance_predicted_labels, 0
    for axiom in relevant_axioms: 
        prompt = f"""You are a logical reasoning validator. Given the axiom and reasoning step:
        1. Axiom : {axiom}
        2. Reasoning step {reasoning_step}

        Your task is to determine the logical relationship:

        -  Output "A" for entailment if the reasoning step logically follows from the axiom or vice versa
        -  Output "B" for contradiction if the reasoning step contradicts the axiom or vice versa
        Just respond with one of the two labels: 'A' or 'B'."""
        prompts.append(prompt)

    # responses = target_client.completions.create(
    #                 model=args.target_model_name_or_path.split("/")[-1],
    #                 prompt=prompts,
    #                 temperature=args.temperature,
    #                 top_p=args.top_p,
    #                 max_tokens=args.max_tokens_per_call,
    #                 stop=[args.step_word],
    #             ).choices

    labels = ['A', 'B']
    
    # target_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True)
    logits_processor = LabelRestrictor(target_tokenizer, labels)

    gen_output = llm.generate(prompts, SamplingParams(
                                    temperature=0, top_p=1,
                                    max_tokens=1,
                                    seed=0,
                                    n=1,
                                    logits_processors=[logits_processor]))
    # print('gen_output', gen_output[0].outputs[0].text)
    # predicted_labels.append(['contradiction', 'entailment', 'neutral'][logits.argmax(-1).item()])
    predicted_labels = [gen_output[index].outputs[0].text for index in range(len(prompts))]
    if ' B' in predicted_labels:
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == ' B':
                print(relevant_axioms[i])
        return relevant_axioms, predicted_labels, 0
    return relevant_axioms, predicted_labels, 1

def faithfulness_check(axioms, reasoning_chain):
    reasoning_list = reasoning_chain.split('.')
    counter = 0
    output_premises, output_predicted_labels, output_sum = [], [], 0
    print('--------------------reasoning chain start-------------------')
    for reason in reasoning_list:
        if reason != '':
            output_premise, output_predicted_labes, output = classify_relevant_reasoning_against_axioms(reason, axioms.split('\n')[1:])
            print('reason', output)
            print('premise', output_premise)
            output_premises.append(output_premise)
            output_predicted_labels.append(output_predicted_labes)
            output_sum += output
            counter += 1
    print('--------------------reasoning chain end-------------------')
    if counter == 0:
        return (output_premises, output_predicted_labels, 0)
    print('faithfulness', output_sum/counter)
    return (output_premises, output_predicted_labels, output_sum/counter)

def correctness_check(real_answer, selected_answer):
    if real_answer == selected_answer:
        return 1
    else:
        return 0

import re
def reasoning_return(output):
    match = re.search(r'### Reasoning:\s*(.*)', output, re.DOTALL)

    if match:
        extracted_text = match.group(1).strip()  # Get matched text and remove leading/trailing spaces
        return extracted_text
    else:
        return ""

def answer_return(output):
    match = re.search(r'### Selected Answer:\s*(.*)', output, re.DOTALL)

    if match:
        extracted_text = match.group(1).strip()  # Get matched text and remove leading/trailing spaces
        return extracted_text
    else:
        return ""

options_list = ["A","B","C","D","E"]

def get_first_capital_letter(text):
    for char in text:
        if char.isupper():
            return char
    return None

import json
with open("/workspace/RSD/external/qwen25_math_evaluation/data/combined_llama70B/combined_llama70B.json", "r") as file:
    data = json.load(file)  

import pandas as pd
df = pd.read_csv("/workspace/RSD/external/qwen25_math_evaluation/data/final_created.csv",index_col=0)
print(len(data), len(df))

from typing_extensions import final
final_output = []
final_dicts = []
for i in range(0,25):
    track_dict = {
        "Persona": {
                "faithfullness_total": 0,
                "correctness_total": 0,
                "counter": 0,
            },
        "Axioms": {
                "faithfullness_total": 0,
                "correctness_total": 0,
                "counter": 0,
            },
        "Persona_and_Axioms": {
            "faithfullness_total": 0,
            "correctness_total": 0,
            "counter": 0,
        },
        "Base": {
            "faithfullness_total": 0,
            "correctness_total": 0,
            "counter": 0,
        }
    }
    temp_dict = {i : {}}
    for key in data[i].keys():
        print(key)
        if ("Persona_" in key) and ("Axioms_" not in key) and ("for" not in key):
            #correctness compute
            temp = {eval(eval(df[key.split("Persona_")[1]][i])['option_mapping'])[j] : options_list[j] for j in range(len(eval(eval(df[key.split("Persona_")[1]][i])['option_mapping'])))}
            real_answer = temp[eval(df[key.split("Persona_")[1]][i])['response']]
            try:
                selected_answer = get_first_capital_letter(data[i][key]['Selected Answer'])
            except:
                selected_answer = ""
                # selected_answer = answer_return(data[i][key]['Selected Answer'])
                track_dict['Persona']['correctness_total'] += correctness_check(real_answer, selected_answer)

            #faithfulness compute
            print(data[i][key]['Reasoning'])
            axioms = data[i]['Axioms_for_'+key.split("Persona_")[1]]
            reasoning_chain = data[i][key]['Reasoning']
            output_resp = faithfulness_check(axioms, reasoning_chain)
            track_dict['Persona']['faithfullness_total'] += output_resp[2]
            temp_dict[i][key+"_Persona"] = [output_resp[0], output_resp[1]]
            track_dict['Persona']['counter'] += 1
        elif ("Axioms_" in key) and ("Persona_" not in key) and ("for" not in key):
            # #correctness compute
            temp = {eval(eval(df[key.split("Axioms_")[1]][i])['option_mapping'])[j] : options_list[j] for j in range(len(eval(eval(df[key.split("Axioms_")[1]][i])['option_mapping'])))}
            real_answer = temp[eval(df[key.split("Axioms_")[1]][i])['response']]
            try:
                selected_answer = get_first_capital_letter(data[i][key]['Selected Answer'])
            except:
                selected_answer = ""
                # selected_answer = answer_return(data[i][key]['Selected Answer'])
                track_dict['Axioms']['correctness_total'] += correctness_check(real_answer, selected_answer)

            #faithfulness compute
            print(data[i][key]['Reasoning'])
            axioms = data[i]['Axioms_for_'+key.split("Axioms_")[1]]
            reasoning_chain = data[i][key]['Reasoning']
            output_resp = faithfulness_check(axioms, reasoning_chain)
            track_dict['Axioms']['faithfullness_total'] += output_resp[2]
            temp_dict[i][key+"_Axioms"] = [output_resp[0], output_resp[1]]
            track_dict['Axioms']['counter'] += 1
        elif ("Axioms_" in key) and ("Persona_" in key) and ("for" not in key):
            # #correctness compute
            temp = {eval(eval(df[key.split("Persona_and_Axioms_")[1]][i])['option_mapping'])[j] : options_list[j] for j in range(len(eval(eval(df[key.split("Persona_and_Axioms_")[1]][i])['option_mapping'])))}
            real_answer = temp[eval(df[key.split("Persona_and_Axioms_")[1]][i])['response']]
            try:
                selected_answer = get_first_capital_letter(data[i][key]['Selected Answer'])
            except:
                selected_answer = ""
                # selected_answer = answer_return(data[i][key]['Selected Answer'])
                track_dict['Persona_and_Axioms']['correctness_total'] += correctness_check(real_answer, selected_answer)

            # #faithfulness compute
            print(data[i][key]['Reasoning'])
            axioms = data[i]['Axioms_for_'+key.split("Persona_and_Axioms_")[1]]
            reasoning_chain = data[i][key]['Reasoning']
            output_resp = faithfulness_check(axioms, reasoning_chain)
            track_dict['Persona_and_Axioms']['faithfullness_total'] += output_resp[2]
            temp_dict[i][key+"_Persona_and_Axioms"] = [output_resp[0], output_resp[1]]
            track_dict['Persona_and_Axioms']['counter'] += 1
        elif ("Base_" in key) and ("for" not in key):
            #correctness compute
            temp = {eval(eval(df[key.split("Base_")[1]][i])['option_mapping'])[j] : options_list[j] for j in range(len(eval(eval(df[key.split("Base_")[1]][i])['option_mapping'])))}
            real_answer = temp[eval(df[key.split("Base_")[1]][i])['response']]
            try:
                selected_answer = get_first_capital_letter(data[i][key]['Selected Answer'])
            except:
                selected_answer = ""
                track_dict['Base']['correctness_total'] += correctness_check(real_answer, selected_answer)

            #faithfulness compute
            # axioms = data[i]['Axioms_for_'+key.split("Base_")[1]]
            # reasoning_chain = data[i][key]['Reasoning']
            # track_dict['Persona_and_Axioms']['faithfullness_total'] += faithfulness_score(faithfulness_check(axioms, reasoning_chain).choices[0].message.content)
            track_dict['Base']['counter'] += 1
    final_dicts.append(temp_dict)
    temp = {}
    for k,v in track_dict.items():
        temp[k] = {'correctness_total': v['correctness_total']/v['counter'], 'faithfulness_value': v['faithfullness_total']/v['counter']}

    print(temp)
    final_output.append(temp)

json.dump(final_dicts, open("combined_llama70B_NLI_relevant_0_to_25.json", "w"), indent=4)
json.dump(final_output, open("combined_llama70B_NLI_relevant_outputs_0_to_25.json", "w"), indent=4)
# if __name__ == "__main__":
    # args = parse_args()
    # set_seed(args.seed)
    # setup(args)
    # print(classify_reasoning_against_axioms("The square of an even number is even.", ["1. If a number is even, then it can be expressed as 2 times an integer. 2. Squaring the number results in 4 times the square of that integer, which is also even."]))
