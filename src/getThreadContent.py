from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()

# my_thread = client.beta.threads.retrieve("thread_MdIN2R0YqjgsPgGAqZ0dhiJq")

thread_messages = client.beta.threads.messages.list("thread_MdIN2R0YqjgsPgGAqZ0dhiJq")
print(thread_messages.data)