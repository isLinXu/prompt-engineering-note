import os
import argparse
from dotenv import load_dotenv, find_dotenv
import openai

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

parser = argparse.ArgumentParser(description='Generate text using OpenAI GPT-3.')
parser.add_argument('--prompt', '-p', required=True, help='The prompt to generate text from')
parser.add_argument('--model', '-m', default='gpt-3.5-turbo', help='The GPT-3 model to use')
args = parser.parse_args()

output = get_completion(args.prompt, args.model)