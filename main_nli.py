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

from external.qwen25_math_evaluation.evaluate import evaluate
from external.qwen25_math_evaluation.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from external.qwen25_math_evaluation.parser import *
from external.qwen25_math_evaluation.trajectory import *
from external.qwen25_math_evaluation.data_loader import load_data
from external.qwen25_math_evaluation.python_executor import PythonExecutor
from external.skywork_o1_prm_inference.model_utils.io_utils import prepare_input, derive_step_rewards_vllm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="math500", type=str)
    parser.add_argument("--data_dir", default="./external/qwen25_math_evaluation/data", type=str)
    parser.add_argument("--draft_model_name_or_path", default="Qwen/Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--draft_model_ip_address", default="http://localhost:12340/v1", type=str)
    parser.add_argument("--target_model_name_or_path", default="Qwen/Qwen2.5-32B-Instruct", type=str)
    parser.add_argument("--target_model_ip_address", default="http://localhost:12341/v1", type=str)
    parser.add_argument("--prm_name_or_path", default="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B", type=str)
    parser.add_argument("--prm_ip_address", default="http://localhost:12342/v1", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="qwen25-math-cot", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--step_word", type=str, default="\n\n")
    parser.add_argument("--prm_threshold", type=float, default=0.7)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    args.fix_type = 'sequential'
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # shuffle
    # if args.shuffle:
    #     random.seed(datetime.now().timestamp())
    #     random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_delta{args.prm_threshold}_maxsteps{args.max_steps}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    # load model
    openai_api_key = "EMPTY"
    # draft_client = OpenAI(
    #     api_key=openai_api_key,
    #     base_url=args.draft_model_ip_address,
    # )
    # draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name_or_path, trust_remote_code=True)

    target_client = OpenAI(
        api_key=openai_api_key,
        base_url=args.target_model_ip_address,
    )
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_name_or_path, trust_remote_code=True)

    # prm_client = OpenAI(
    #     api_key=openai_api_key,
    #     base_url=args.prm_ip_address,
    # )
    # prm_tokenizer = AutoTokenizer.from_pretrained(args.prm_name_or_path, trust_remote_code=True)

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(target_client, target_tokenizer, data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append({"acc": sum([result["acc"] for result in results]) / len(results),})

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

# def sort_all_responses(current_prompts, responses, k):
#     if len(current_prompts) == len(responses):
#         responses = sorted(responses, key=lambda x: int(x.index))
#     else:
#     sorted_responses = []
#     for (orig_idx, p, responses_set) in current_prompts:
#         if len(responses_set[-1])>1:
            

#         idx = int(responses[i].index)
#         sorted_responses.append((idx, responses[i]))
#     sorted_responses = [resp for idx, resp in sorted(sorted_responses, key=lambda x: x[0])]
#     return sorted_responses


def get_responses(args, target_client, target_tokenizer, prompts, problems, mode='single'):
    final_outputs = [None] * len(prompts) 
    outputs = {idx : {} for idx in range(len(prompts))} # Initialize with None for tracking
    token_counts = [(0, 0, 0) for _ in prompts]  # (draft_tokens, target_tokens, discarded_draft_tokens) for each prompt
    step_info = [[] for _ in prompts]  # List to store (step_num, client_id) for each prompt
    reward_track = {idx : {} for idx in range(len(prompts))}  # track the reward for each prompt
    current_prompts = [(i, -1, p, []) for i, p in enumerate(prompts)] # (index, reference index, prompt, responses)
    all_rewards = [[] for _ in prompts]   # List to store (step_num, client_id) for each prompt
    current_problems = problems
    num_step = 0
    pre_num_finished = 0
    num_unchanged = 0
    k = 1 if mode == 'single' else 3
   
    while current_prompts:
        #flatten the current_prompts
        batch_prompts = []
        batch_prompts = [p + ''.join(r[0] for r in responses) for _, _, p, responses in current_prompts]
        # Firstly generate with the draft model
        draft_responses = draft_client.completions.create(
            model=args.draft_model_name_or_path.split("/")[-1],
            prompt=batch_prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_per_call,
            stop=[args.step_word],
        ).choices
        draft_responses = sorted(draft_responses, key=lambda x: int(x.index))

        #flatten with reference index
        # current_prompts_prm = []
        # current_problems_prm = []
        # for (orig_idx, ref_idx, p, prev_resp) in current_prompts:
        #     current_prompts_prm.append((orig_idx, ref_idx, p, prev_resp))
        #     current_problems_prm.append(current_problems[orig_idx])
        # print("current_prompts_prm", current_prompts_prm)
        full_responses = [''.join(r[0] for r in prev_resp) + new_resp.text
                    for (_, _, _, prev_resp), new_resp in zip(current_prompts, draft_responses)]
        processed_data = [
            prepare_input(p, full_resp, tokenizer=prm_tokenizer, step_token=args.step_word) 
            for p, full_resp in zip(current_problems, full_responses)
        ]
        input_ids, steps, reward_flags = zip(*processed_data)
        rewards = prm_client.embeddings.create(
            input=input_ids,
            model=args.prm_name_or_path.split("/")[-1],
        )
        step_rewards = derive_step_rewards_vllm(rewards, reward_flags) # list[list]

        # Split prompts based on step_reward
        good_prompts = []
        bad_prompts = []
        # print("current_prompts", current_prompts)
        # print('reward_track', reward_track)
        # print('step_rewards', step_rewards)
        for current_idx, ((orig_idx, ref_idx, prompt, prev_responses), draft_response, step_reward) in enumerate(zip(current_prompts, draft_responses, step_rewards)):
            # all_rewards[current_idx].append(round(step_reward[-1], 6))
            if step_reward[-1] >= args.prm_threshold:
                good_prompts.append((orig_idx, ref_idx, prompt, prev_responses, draft_response, True))  # True means using draft model
                if ref_idx == -1 and ref_idx not in reward_track[orig_idx]:
                    reward_track[orig_idx][ref_idx] = [round(step_reward[-1], 6)]
                else:
                    reward_track[orig_idx][ref_idx].append(round(step_reward[-1], 6))
            elif step_reward[-1] < args.prm_threshold and ref_idx != -1:
                good_prompts.append((orig_idx, ref_idx, prompt, prev_responses, draft_response, False))  
                reward_track[orig_idx][ref_idx].append(round(step_reward[-1], 6))
            else:
                draft_response_text = draft_response.text + args.step_word
                token_counts[orig_idx] = (
                    token_counts[orig_idx][0], 
                    token_counts[orig_idx][1], 
                    token_counts[orig_idx][2]+len(draft_tokenizer.encode(draft_response_text))
                )
                bad_prompts.append((orig_idx, ref_idx, prompt, prev_responses))

        print(f"Good prompts: {len(good_prompts)}, Bad prompts: {len(bad_prompts)}")
        # Generate using target model for bad prompts
        if bad_prompts:
            batch_prompts_parallel = [p + ''.join(r[0] for r in responses) for _, _, p, responses in bad_prompts]
            ladder = [(0.6, 0.90), (0.8, 0.92), (1.0, 0.95)]
            branches = []
            for t,tp in ladder:
                target_responses = draft_client.completions.create(
                    model=args.draft_model_name_or_path.split("/")[-1],
                    prompt=batch_prompts_parallel,
                    temperature=t,
                    top_p=tp,
                    max_tokens=args.max_tokens_per_call, 
                    n=1,
                    stop=[args.step_word],
                ).choices
                branches.extend(target_responses)
            assert len(branches) == len(bad_prompts)*3
            target_responses = sorted(branches, key=lambda x: int(x.index))
            target_responses = [target_responses[i:i+3] for i in range(0, len(target_responses), 3)]
            for (orig_idx, ref_idx, prompt, prev_responses), target_response in zip(bad_prompts, target_responses):
                for t_r in range(len(target_response)):
                    if ref_idx not in reward_track[orig_idx].keys():
                        reward_track[orig_idx][t_r] = []
                    else:
                        reward_track[orig_idx][t_r] = reward_track[orig_idx][ref_idx].copy()
                    good_prompts.append((orig_idx, t_r, prompt, prev_responses, target_response[t_r], False))  # False means using target model
                        
                # # Add target model responses to good_prompts
                # for (orig_idx, prompt, prev_responses), target_response in zip(bad_prompts, target_responses):
                #     good_prompts.append((orig_idx, prompt, prev_responses, target_response, False))  # False means using target model
        
        # Process all responses
        next_prompts = []
        next_problems = []
        temp_responses = {orig_idx: [] for orig_idx, _, _, _ in current_prompts}
        temp_ref_idx = {orig_idx: [] for orig_idx, _, _, _ in current_prompts}
        for orig_idx, ref_idx, prompt, prev_responses, response, used_draft in sorted(good_prompts, key=lambda x: x[0]):
            response_text = response.text + args.step_word
            client_id = 1 if used_draft else 2
            tokenizer = draft_tokenizer
            num_tokens = len(tokenizer.encode(response_text))
            
            # Update token counts
            if client_id == 1:
                token_counts[orig_idx] = (token_counts[orig_idx][0] + num_tokens, token_counts[orig_idx][1], token_counts[orig_idx][2])
            else:
                token_counts[orig_idx] = (token_counts[orig_idx][0], token_counts[orig_idx][1] + num_tokens, token_counts[orig_idx][2])
            
            # Record step information
            step_info[orig_idx].append((num_step, client_id))

            full_responses = prev_responses + [(response_text, client_id)]
            full_responses_text = ''.join(r[0] for r in full_responses)
            # temp_responses[orig_idx].append(full_responses)
            # temp_ref_idx[orig_idx].append(ref_idx)
            # terminate conditions
            if (response.stop_reason is None) \
             or len(draft_tokenizer.encode(prompt + full_responses_text)) >= args.max_tokens_per_call \
             or num_step >= args.max_steps - 1 \
             or num_unchanged >= args.patience - 1:
                outputs[orig_idx][ref_idx] = full_responses_text[:-len(args.step_word)]
            else:
                next_prompts.append((orig_idx, ref_idx, prompt, full_responses))
                next_problems.append(problems[orig_idx])
        
        current_prompts = next_prompts
        current_problems = next_problems

        assert len(current_prompts) == len(current_problems)
        current_prompts_in_rotation = []
        for orig_idx, ref_idx, prompt, full_responses in current_prompts:
            if orig_idx not in current_prompts_in_rotation:
                current_prompts_in_rotation.append(orig_idx)
        if len(outputs) - len(current_prompts_in_rotation) > pre_num_finished:
            num_unchanged = 0
            pre_num_finished = len(outputs) - len(current_prompts_in_rotation)
        else:
            num_unchanged += 1

        print(f"#### Step {num_step}: Completed {pre_num_finished} / {len(outputs)}, #unchanged {num_unchanged} / {args.patience}")
        num_step += 1

    # select the response with highest cumulative reward
    
    for orig_idx, value in outputs.items():
        if len(value.keys()) > 1:
            if -1 in reward_track[orig_idx]:
                del reward_track[orig_idx][-1]
            print("reward_track[orig_idx]", reward_track[orig_idx], value)
            totals = {ki: sum(reward_track[orig_idx][ki])/len(reward_track[orig_idx][ki]) for ki, v in value.items() if reward_track[orig_idx][ki] != []}
            max_key = max(totals, key=totals.get)
            final_outputs[orig_idx] = outputs[orig_idx][max_key]
            print('max_key', max_key, "orig", orig_idx, 'reward_track', reward_track[orig_idx])
            all_rewards[orig_idx] = reward_track[orig_idx][max_key]
        else:
            final_outputs[orig_idx] = outputs[orig_idx][-1]
            all_rewards[orig_idx] = reward_track[orig_idx][-1]

    return final_outputs, token_counts, step_info, all_rewards


def main(target_client, target_tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            draft_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # start inference
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        problems = [sample["question"] for sample in samples]
        assert len(prompts) == len(problems)
        outputs, token_counts, turn_info, all_rewards = get_responses(
            args,
            target_client,
            target_tokenizer,
            prompts,
            problems,
        )
        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update(
            {"code": code, "pred": preds, "report": reports, 
             "token_counts": token_counts[i], "turn_info": turn_info[i], "reward": all_rewards[i]}
        )
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    # save metrics
    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    llm1_tokens = [0, 0] # (correct, wrong)
    llm1_discarded_tokens = [0, 0]
    llm2_tokens = [0, 0]
    for i, sample in enumerate(all_samples):
        if sample["score"][0]:
            llm1_tokens[0] += sample["token_counts"][0]
            llm2_tokens[0] += sample["token_counts"][1]
            llm1_discarded_tokens[0] += sample["token_counts"][2]
        else:
            llm1_tokens[1] += sample["token_counts"][0]
            llm2_tokens[1] += sample["token_counts"][1]
            llm1_discarded_tokens[1] += sample["token_counts"][2]
    total_tokens = sum(llm1_tokens) + sum(llm2_tokens) + sum(llm1_discarded_tokens)
    total_tokens_for_correct_pred = llm1_discarded_tokens[0] + llm1_tokens[0] + llm2_tokens[0]
    total_tokens_for_wrong_pred = llm1_discarded_tokens[1] + llm1_tokens[1] + llm2_tokens[1]

    result_json["tokens_ratio_overall(llm1,llm2)"] = (
        (sum(llm1_tokens)+sum(llm1_discarded_tokens))/total_tokens, sum(llm2_tokens)/total_tokens
    ) if total_tokens > 0 else (0,0) 
    result_json["tokens_ratio_correct_prediction(llm1,llm2)"] = (
        (llm1_discarded_tokens[0]+llm1_tokens[0])/total_tokens_for_correct_pred, llm2_tokens[0]/total_tokens_for_correct_pred
    ) if total_tokens_for_correct_pred > 0 else (0,0) 
    result_json["tokens_ratio_wrong_prediction(llm1,llm2)"] = (
        (llm1_discarded_tokens[1]+llm1_tokens[1])/total_tokens_for_wrong_pred, llm2_tokens[1]/total_tokens_for_wrong_pred
    ) if total_tokens_for_wrong_pred > 0 else (0,0) 
    result_json["tokens_ratio(correct,wrong)"] = (
        total_tokens_for_correct_pred/total_tokens, total_tokens_for_wrong_pred/total_tokens
    ) if total_tokens > 0 else (0,0) 
    result_json["tokens_ratio_discarded(correct,wrong)"] = (
        llm1_discarded_tokens[0]/total_tokens_for_correct_pred, llm1_discarded_tokens[1]/total_tokens_for_wrong_pred
    ) if (total_tokens_for_correct_pred > 0 and total_tokens_for_wrong_pred > 0)  else (0,0) 
    result_json["acceptance_rate"] = (
        (llm1_tokens[0] + llm1_tokens[1])/(llm1_tokens[0] + llm1_tokens[1] + llm1_discarded_tokens[0] + llm1_discarded_tokens[1])
    ) if ((llm1_tokens[0] + llm1_tokens[1]) > 0)  else 0
    result_json["num_draft_tokens"] = sum(llm1_tokens) + sum(llm1_discarded_tokens)
    result_json["num_target_tokens"] = sum(llm2_tokens)

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json

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
