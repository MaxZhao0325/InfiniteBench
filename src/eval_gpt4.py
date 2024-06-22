from openai import OpenAI
import tiktoken
from pathlib import Path
from eval_utils import (
    create_msgs,
    load_data,
    dump_jsonl,
    iter_jsonl,
    get_answer,
)
import time
from args import parse_args
from dotenv import load_dotenv
load_dotenv()

api_key = ""
org_id = ""


# client = OpenAI(
#     api_key=api_key,
#     organization=org_id,
# )

client = OpenAI()

# assistant = client.beta.assistants.create(
#   name="Assistant",
#   instructions="You are a personal math tutor. Write and run code to answer math questions.",
#   model="gpt-4o",
# )

assistant = client.beta.assistants.create(
        name="Assistant_longbook_choice_eng",
        model="gpt-4o",
    )


def chat(messages: list, input='', op1='', op2='', op3='', op4=''):   
    instruction = messages[0]["content"]
    new_messages = []
    original_messages = messages[1]["content"]
    while(original_messages):
        cut_message=original_messages[:256000]
        new_messages.append({"role": "user", "content": cut_message})
        original_messages = original_messages[256000:]
    # if(input):
    #     input_string = "问题：{question}\n请尽量简短地回答。"
    #     input_string = input_string.format(
    #         question=input,
    #     )
    #     input_message = {"role": "user", "content": input_string}
    #     new_messages.append(input_message)
    if(input and op1 and op2 and op3 and op4):
        input_string = "Question: {question}\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}"
        input_string = input_string.format(
            question=input,
            OPTION_A=op1,
            OPTION_B=op2,
            OPTION_C=op3,
            OPTION_D=op4,
        )
        input_message = {"role": "user", "content": input_string}
        print("input", input_string)
        new_messages.append(input_message)
        
    thread = client.beta.threads.create(messages=new_messages)
    print(thread.id)
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions=instruction,
        max_prompt_tokens = 5000
    )
    
    if run.status == 'completed': 
        messages = client.beta.threads.messages.list(
            thread_id=thread.id, order="desc"
        )
        print(messages.data[0].content[0].text.value)
        return messages.data[0].content[0].text.value
    else:
        return ""


if __name__ == "__main__":

    args = parse_args()
    verbose = args.verbose
    task = args.task

    examples = load_data(task)

    result_dir = Path(args.output_dir)
    result_dir.mkdir(exist_ok=True, parents=True)

    output_path = result_dir / "assistant" / f"preds_{task}2.jsonl"
    if output_path.exists():
        preds = list(iter_jsonl(output_path))
        start_idx = len(preds)
        stop_idx = len(examples)
    else:
        start_idx = 0
        stop_idx = len(examples)
        preds = []
    # tokenizer = tiktoken.encoding_for_model("gpt-4")

    start_time = time.time()
    # i = start_idx
    i=0
    while i < stop_idx and i<1:
        eg = examples[i]
        msgs, prompt = create_msgs(
            eg, task, model_name="gpt4", data_dir=args.data_dir
        )
        if verbose:
            print(f"======== Example {i} =========")
            print("Input text:")
            print(prompt[:300])
            print("...")
            print(prompt[-300:])
            print("==============================")
        # Make prediction
        try:
            # response = chat(msgs, eg["input"], eg["options"][0], eg["options"][1], eg["options"][2], eg["options"][3])
            response = chat(msgs)
            preds.append(
                {
                    "id": i,
                    "prediction": response,
                    "ground_truth": get_answer(eg, task),
                }
            )
            # Save result
            dump_jsonl(preds, output_path)
            print("Time spent:", round(time.time() - start_time))
            # exit()
            print(response)
            time.sleep(20)
            i += 1
        except Exception as e:
            print("ERROR:", e)
            print("Retrying...")
            time.sleep(60)