import requests
import pandas as pd
import datetime
import os
import traceback
import re
from openai import OpenAI
from google import genai
from google.genai import types
import anthropic
import time

"""   
INFO ABOUT THIS SOURCE CODE

This source code provides functions to aggregate conversation with the LLM during the multi-turn attacks.
The main functions included are:

1. conversation_lm_studio(): Handles chat conversations with a large language model (LLM) via the LM Studio server.
2. conversation_chatgpt(): Handles chat conversations with ChatGPT of OpenAI via API.
3. conversation_gemini(): Handles chat conversations with Gemini of Google via API.
4. conversation_claude(): Handles chat conversations with Claude of Anthropic via API.
5. process_conversation(): Handles new chat conversation according to the selected LLM.


"""


# Define the API endpoint and headers (get them all from LM studio when enabling the Server mode)
API_URL = "<Insert LM Studio's Server API endpoint here...>"  # LM Studio's API endpoint
HEADERS = {"Content-Type": "application/json"}

# Max tokens parameter set to 1024 during the experiments
MAX_TOKENS = 1024

# Top parameter
TOP_P = 0.9


class LLMAggregator:

    # The input file is the baseline dataset (AdvBench+ HarmBench)
    # It should be in the same folder with this code or the path to the file should be modified accordingly
    INPUT_FILE = 'Baseline_Dataset_Advbench_HarmBench.csv'
    
    
    # Dataset folder name
    DATASET_FOLDER = 'datasets/'

    # Test folder
    TEST_FOLDER = 'tests/'

    # Dataset dictionary
    MULTITURN_DATASETS = {
        "2": DATASET_FOLDER + "CyMulTenSet_N_2.csv",
        "3": DATASET_FOLDER + "CyMulTenSet_N_3.csv"
        }
    
    # List of the opensource LLMs we consider during our experiments, their names can be retrieved from LM studio when the model is loaded
    # LLM 1: llama-2-7b-chat
    # LLM 2: qwen2-7b-instruct
    # LLM 3: mistral-7b-instruct-v0.1
    OPEN_LLM_VERSIONS = [
        {"MODEL_NAME": "llama-2-7b-chat", "LLM": "Llama", "LLM_VERSION": "2 (7B)"},
        {"MODEL_NAME": "qwen2-7b-instruct", "LLM": "Qwen", "LLM_VERSION": "2 (7B)"},
        {"MODEL_NAME": "mistral-7b-instruct-v0.1", "LLM": "Mistral", "LLM_VERSION": "0.1 (7B)"}
    ]

    # Define the API keys of the close source LLMs
    GPT4_API_KEY = "<Insert API here...>"
    GEMINI_API_KEY = "<Insert API here...>"
    CLAUDE_API_KEY = "<Insert API here...>"
    
    # List of the close source LLMs we consider during our experiments, their names can be retrieved from LM studio when the model is loaded
    # LLM 1: gpt-4o-mini
    # LLM 2: gemini-2.0-flash
    # LLM 3: claude-3-5-sonnet-20241022
    CLOSE_LLM_VERSIONS = [
        {"MODEL_NAME": "gpt-4o-mini", "LLM": "GPT", "LLM_VERSION": "4o (mini)"},
        {"MODEL_NAME": "gemini-2.0-flash", "LLM": "Gemini", "LLM_VERSION": "2.0 (Flash)"},
        {"MODEL_NAME": "claude-3-5-sonnet-20241022", "LLM": "Claude", "LLM_VERSION": "3.5 (Sonnet)"}
    ]

    # List the API keys in the same order as in the close-source LLMs' list
    API_KEYS = [GPT4_API_KEY, GEMINI_API_KEY, CLAUDE_API_KEY]

    # List of different variations of hyperparameters temperature we use during the experiments: 0.1, 0.5 and 0.9
    TEMPS = [0.1, 0.5, 0.9]

    # Function that control and process conversations depending on the chosen LLM, wether close or open source
    # Input:
    # is_selected_llm_open: True if the selected LLM is open-source, false otherwise
    # index: The index of the selected LLM in the list
    # temperature: The value of the hyperparameter temperature of the LLM during the experiment
    # conversations: Existing conversation to use and to append new text before querying the targeted LLM
    # newtext: The new text to add to the conversation before querying the LLM
    # Output:
    # This function returns the response of the LLM as a string
    @staticmethod
    def process_conversation(is_selected_llm_open, index, temperature, conversations, newtext):
        """
        Processes a conversation based on is_open and index values.
        Appends the new user input to the conversations list and 
        calls the appropriate conversation function depending on the selected model.
        """
    
        # If an open-source LLM is selected
        if is_selected_llm_open:
            conversations.append({
                "role": "user",
                "content": newtext
            })
            model_name = LLMAggregator.OPEN_LLM_VERSIONS[index]["MODEL_NAME"]
            return LLMAggregator.conversation_lm_studio(conversations, model_name, temperature)
    
        # If a closed-source LLM is selected
        model_name = LLMAggregator.CLOSE_LLM_VERSIONS[index]["MODEL_NAME"]
    
        if index == 1:  # Gemini
            conversations.append({
                "role": "user",
                "parts": [{"text": newtext}]
            })
            return LLMAggregator.conversation_gemini(conversations, model_name, temperature)
    
        elif index == 2:  # Claude
            conversations.append({
                "role": "user",
                "content": newtext
            })
            return LLMAggregator.conversation_claude(conversations, model_name, temperature)
    
        else:  # Default: ChatGPT (index == 0 or fallback)
            conversations.append({
                "role": "user",
                "content": newtext
            })
            return LLMAggregator.conversation_chatgpt(conversations, model_name, temperature)
            


    # Function of chat conversations with the LLM via LM Studio Server
    # We minimize the randomness of the model's predictions to make it more deterministic in the response by setting temperature to a lower value
    # We Maximize diversity in the model’s responses potentially including highly improbable tokens by setting top_p to a higher value
    # We are limiting the max_tokens to 1024
    # Input:
    # conversation: The conversation we want to input to the LLM for a response.
    # model_name: The name of the LLM in LM studio that we want to use
    # max_tokens: The limit of tokens the LLM should use during the response generation process. It's set by default to MAX_TOKENS value.
    # temperature: The value of the hyperparameter temperature of the LLM during the experiment
    # top_p: The higher is this value the more diverse is the model’s responses and conversely. It's set by default to TOP_P value.
    # Output:
    # This function returns the response of the LLM as a string
    @staticmethod
    def conversation_lm_studio(conversation, model_name, temperature, max_tokens=MAX_TOKENS, top_p=TOP_P):
        """
        Query a specific AI model running on LM Studio.
    
        Args:
            message (list[dict]): A list of dictionaries where each dictionary contains:
                - "role" (str): The role of the message sender, e.g., "user" or "assistant".
                - "content" (str): The content of the message, which could be the prompt or the assistant's reply.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature for randomness in responses.
            top_p (float): Nucleus sampling for diversity in responses.
            model_name (str): The name of the specific model to use (optional, depends on LM Studio's configuration).
    
        Returns:
            str: The response generated by the AI model.
        """
        # Prepare the payload
        payload = {
            "messages": conversation,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "model": model_name
        }
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload)
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response text found.")
        except requests.exceptions.RequestException as e:
            return f"Error querying LM Studio: {e}"




    # Function of chat conversations with ChatGPT via API
    # We minimize the randomness of the model's predictions to make it more deterministic in the response by setting temperature to a lower value
    # We Maximize diversity in the model’s responses potentially including highly improbable tokens by setting top_p to a higher value
    # We are limiting the max_tokens to 1024
    # Input:
    # conversation: The conversation we want to input to the LLM for a response.
    # model_name: The name of the LLM (in the OpenAI API Documentation) that we want to use
    # max_tokens: The limit of tokens the LLM should use during the response generation process. It's set by default to MAX_TOKENS value.
    # temperature: The value of the hyperparameter temperature of the LLM during the experiment
    # top_p: The higher is this value the more diverse is the model’s responses and conversely. It's set by default to TOP_P value.
    # Output:
    # This function returns the response of the LLM as a string
    @staticmethod
    def conversation_chatgpt(conversation, model_name, temperature, max_tokens=MAX_TOKENS, top_p=TOP_P):
        try:
            client = OpenAI(
              api_key=os.environ["API_KEY"]
            )
            
            response = client.chat.completions.create(
              model=model_name,
              max_tokens=max_tokens,
              temperature=temperature,
              store=False,
              messages=conversation
            )
            # Check if there is at least one choice in the response
            if response.choices:
                return response.choices[0].message.content
            else:
                return "I cannot support."
        except Exception as e:
            # Handle potential errors such as API errors or connection issues
            return f"An error occurred: {str(e)}"
    
    
    # Function of chat conversations with Gemini via API
    # We minimize the randomness of the model's predictions to make it more deterministic in the response by setting temperature to a lower value
    # We Maximize diversity in the model’s responses potentially including highly improbable tokens by setting top_p to a higher value
    # We are limiting the max_tokens to 1024
    # Input:
    # conversation: The conversation we want to input to the LLM for a response.
    # model_name: The name of the LLM (in the Google API Documentation) that we want to use
    # max_tokens: The limit of tokens the LLM should use during the response generation process. It's set by default to MAX_TOKENS value.
    # temperature: The value of the hyperparameter temperature of the LLM during the experiment
    # top_p: The higher is this value the more diverse is the model’s responses and conversely. It's set by default to TOP_P value.
    # Output:
    # This function returns the response of the LLM as a string
    @staticmethod
    def conversation_gemini(conversation, model_name, temperature, max_tokens=MAX_TOKENS, top_p=TOP_P):
        try:
            client = genai.Client(api_key=os.environ["API_KEY"])
            response = client.models.generate_content(
            model=model_name,
            contents=conversation,
            config=types.GenerateContentConfig(
                    temperature = temperature,
                    top_p = top_p,
                    top_k = 0,
                    max_output_tokens = max_tokens
                    )
            )
            #print(response.text)
            # Check if the response is valid and contains text
            if response and response.text:
                return response.text
            else:
                return "I cannot support."
        except Exception as e:
            # Handle potential errors such as API errors or connection issues
            return f"An error occurred: {str(e)}"
    
    # Function of chat conversations with Claude via API
    # We minimize the randomness of the model's predictions to make it more deterministic in the response by setting temperature to a lower value
    # We Maximize diversity in the model’s responses potentially including highly improbable tokens by setting top_p to a higher value
    # We are limiting the max_tokens to 1024
    # Input:
    # conversation: The conversation we want to input to the LLM for a response.
    # model_name: The name of the LLM (in the Anthropic API Documentation) that we want to use
    # max_tokens: The limit of tokens the LLM should use during the response generation process. It's set by default to MAX_TOKENS value.
    # temperature: The value of the hyperparameter temperature of the LLM during the experiment
    # top_p: The higher is this value the more diverse is the model’s responses and conversely. It's set by default to TOP_P value.
    # Output:
    # This function returns the response of the LLM as a string
    @staticmethod
    def conversation_claude(conversation, model_name, temperature, max_tokens=MAX_TOKENS, top_p=TOP_P):
        try:
            client = anthropic.Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key=os.environ["API_KEY"],
            )
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=conversation
            )
            # Check if there is at least one choice in the response
            if response:
                return response.content[0].text
            else:
                return "I cannot support."
        except Exception as e:
            # Handle potential errors such as API errors or connection issues
            return f"An error occurred: {str(e)}"



