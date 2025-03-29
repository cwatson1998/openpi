from google import genai
from dotenv import load_dotenv
import PIL.Image
import os
import time
DEFAULT_MODEL = 'gemini-2.0-flash'

TRUE_WORDS = ['true', 'yes']
FALSE_WORDS = ['false', 'no']

with open('decomposition/prompts/hi-robot-lite.txt') as f:
    hi_robot_lite_prompt = f.read()

class RetryWrapper:
    def __init__(self, base_object, max_retries=16, retry_latency=0.1):
        self.base_object = base_object
        self.max_retries = max_retries
        self.retry_latency = retry_latency

    def __getattr__(self, name):
        attr = getattr(self.base_object, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                time.sleep(1)
                last_exception = None
                for attempt in range(self.max_retries):
                    try:
                        print(f"debug, attempt {attempt}")
                        return attr(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        time.sleep(self.retry_latency * (2**attempt))
                raise last_exception
            return wrapper
        return attr



    

def get_client(google_api_key):
    client = genai.Client(api_key=google_api_key)
    return RetryWrapper(client)


    

def hello_world(client, model=DEFAULT_MODEL):
    response = client.models.generate_content(model=model, contents="Say a hello world message")
    return response.text

def text_to_bool(text, malformed_behavior='error'):
    new_text = text.lower().strip()
    for w in TRUE_WORDS:
        if w in new_text:
            return True
    for w in FALSE_WORDS:
        if w in new_text:
            return False
    if malformed_behavior == 'False':
        return False
    elif malformed_behavior == 'True':
        return True
    else:
        raise ValueError(f'"{text}" is not semantically boolean')

'''def text_to_bool(text, malformed_behavior='error'):
    new_text = text.lower().strip()
    if new_text in ['true', 'yes']:
        return True
    elif new_text in ['no', 'false']:
        return False
    elif malformed_behavior == 'False':
        return False
    elif malformed_behavior == 'True':
        return True
    else:
        raise ValueError(f'"{text}" is not semantically boolean')'''
    
def boolean_vqa(client, image, question, model=DEFAULT_MODEL, malformed_behavior='error'):
    response = client.models.generate_content(
        model=model,
        contents = [image, question, "Please provide a Boolean response (True or False)."]
    )
    return text_to_bool(response.text, malformed_behavior=malformed_behavior)

def detect_success(client, image, action_prompt, **kwargs):
    question = f"Did the robot successfully complete the following task: '{action_prompt}'?"
    return boolean_vqa(client, image, question, **kwargs)

def hi_robot_stateless(client, image, task_description, model=DEFAULT_MODEL, quiet=False):
    response = client.models.generate_content(
        model=model,
        contents = [hi_robot_lite_prompt, "Input image:", image, "Input task description: ", task_description]
    )
    text = response.text.strip().strip('"').strip("'")
    text = text.removeprefix("Gemini: ")
    text = text.removeprefix("Gemini:")

    if not quiet:
        print(f"HiRobot says: {text}")
    return text
