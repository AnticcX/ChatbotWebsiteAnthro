from openai import OpenAI
from tenacity import retry, stop_after_attempt, RetryError

import json, os

class Model:
    """
    Wrapper class for interacting with an OpenAI-compatible LLM model.
    """
    
    MAX_CONVERSATION_LENGTH = 30
    
    def __init__(self, model_base_url: str, model: str, api_key: str, is_vision: bool = False):
        """
        Initializes the model configuration.

        Args:
            model_base_url (str): The base URL for the OpenAI or compatible API.
            model (str): The model ID or name (e.g. 'gpt-4', 'gpt-3.5-turbo').
            api_key (str): The API key used for authentication.
            is_vision (bool, optional): Indicates if the model supports image input. Defaults to False.
        """
        self.model_base_url: str = model_base_url
        self.model: str = model
        self.api_key: str = api_key
        self.is_vision: bool = is_vision 
    
    @staticmethod
    def get_prompt(persona: str) -> str:
        return open(f"./prompts/{persona}.txt", "r", encoding="utf-8").read()

    @staticmethod
    def get_chat(user_id: str, group: str) -> list:
        # Ensure the chats directory exists
        os.makedirs(f'./chats/{user_id}', exist_ok=True)
        path = f'./chats/{user_id}/{group}.json'

        # If file doesn't exist, create it with empty dictionary
        if not os.path.exists(path):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=4)

        # Read and return the content
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data[-20:] if isinstance(data, list) else []
    
    @staticmethod
    def save_chat(user_id: str, group: str, conversation: dict) -> None:
        path = f'./chats/{user_id}/{group}.json'
        with open(path, "w", encoding='utf-8') as f:
            json.dump(conversation, f, indent=2)
            
    def get_client(self) -> OpenAI:
        """
        Creates and returns an OpenAI client instance.

        Returns:
            OpenAI: An instance of the OpenAI client configured with base URL and API key.
        """
        return OpenAI(base_url=self.model_base_url, api_key=self.api_key)
    
    @retry(stop=stop_after_attempt(3))
    def call_chat_completion(self, content: list) -> str:
        client = self.get_client()
        response = client.chat.completions.create(model=self.model, messages=content, temperature=0.85)
        content = response.choices[0].message.content
        return content
    
    def fetch_response(self, user_id: str, content: str, group: str) -> dict:
        conversation = self.get_chat(user_id, group)[-self.MAX_CONVERSATION_LENGTH:]
        if len(conversation) == 0 or conversation[0]["role"] != "system":
            conversation.insert(0, {
                "role": "system",
                "content": self.get_prompt(group)
            })
            
        conversation.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": content
                    }
                ]
            })
        print('\n'.join([i['role'] for i in conversation]))
        try:
            model_response = self.call_chat_completion(list(conversation))
        except RetryError:
            model_response = "  ||  "
        
        conversation.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": model_response
                    }
                ]
            })
        self.save_chat(user_id, group, conversation)
        return model_response