from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
load_dotenv()

client = OpenAI()

# Example messages
messages = [
    {"role": "user", "content": "my name is Hanz"},
    {"role": "user", "content": "my age is 18"},
    {"role": "user", "content": "my gender is male"},
    {"role": "user", "content": "my hobby is football"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my height is 186cm, my weight is 136lbs, my eye color is black"},
    {"role": "user", "content": "my job is software engineer"},
    {"role": "user", "content": "what is my age?"} # cannot find this info since it's been trimmed
    # {"role": "user", "content": "what is my eye color?"} # return correct answer
]

# we first use this function to calculate the total number of tokens for the entire message lists
# and the result is 525 for this example messages
def num_tokens_from_messages(messages, model="gpt-4"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
# the result is 525 for this example messages
# print(num_tokens_from_messages(messages))

# then we test if openai assistant will only retrieve the relevant message to put into context
# and whether it knows my age given limited context window
thread = client.beta.threads.create(messages=messages)

assistant = client.beta.assistants.retrieve(
    assistant_id="asst_oVyJPUmM1uJFHecofgCN3skO"
)
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="You are a helpful assistant.",
    # since the entire token number is 525, i limit the context window to 256 to see if 
    # the assistant will trim the middle part and keep head and tail
    # if it knows my age, then it indeed cut the middle message.
    max_prompt_tokens = 256,
    truncation_strategy={
        "type": "auto",
        "last_messages": None
    },
)

messages = client.beta.threads.messages.list(
    thread_id=thread.id, order="desc"
)
# the result is that the openai assistant does not know my age
# which shows that it just trims out the front messages
print(messages.data[0].content[0].text.value)

# here it shows the usage token of the run
# if we don't have any max_prompt_tokens limit, then the context used is 538 tokens, which is close
# to the 525 we calculated using tiktoken. So, the assistant will not do RAG before it invokes the model.
# the assistant just put all messages within the context to the model.

# if you limit the max_prompt_tokens to 256, then the context used is 235 token, which aligns in the window
print(run.usage)
