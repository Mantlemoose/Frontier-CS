from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict
from openai import OpenAI, APITimeoutError
import google.generativeai as genai
import os
from dotenv import load_dotenv
import anthropic
from anthropic import Anthropic, APITimeoutError as AnthropicAPITimeoutError
from google import genai
from google.genai import types

load_dotenv()

class LLMInterface(ABC):
    """
    Abstract base class for integrating Large Language Models (LLMs) into a competitive programming context.
    """

    def __init__(self):
        """
        Initialize the LLMInterface with a predefined prompt for generating competitive programming solutions.
        """
        self.prompt = """
        You are a competitive programmer. You will be given a problem statement, please implement a solution in C++. The execution time and memory limit are also stated in the statement so be aware of the complexity of the program. Please wrap the code in ```cpp and ``` so that it is properly formatted. Your response should ONLY contain the C++ code, with no additional explanation or text.
        """

    @abstractmethod
    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        """
        Abstract method to interact with the LLM.
        """
        pass

    def generate_solution(self, problem_statement: str) -> Tuple[str, Any]:
        """
        Generates a solution to a given competitive programming problem using the LLM.
        """
        user_prompt = self.prompt + problem_statement
        response, meta = self.call_llm(user_prompt)
        return response, meta


class GPT(LLMInterface):
    """
    Concrete implementation of LLMInterface using OpenAI's GPT-4o model.
    """

    def __init__(self):
        """
        Initializes the ExampleLLM class by creating an instance of the OpenAI client.
        """
        super().__init__()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.name = 'gpt'

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        """
        Sends the user prompt to OpenAI's GPT-5 model and retrieves the solution.

        Args:
            user_prompt (str): The complete prompt including the initial context and problem statement.

        Returns:
            Tuple[str, Any]: The LLM's response and metadata about the completion.
        """
        try:
            completion = self.client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": user_prompt}],
                reasoning_effort="high",
                timeout=1200.0
            )
            return completion.choices[0].message.content, str(completion)
        except APITimeoutError as e:
            print(f"OpenAI API request timed out: {e}")
            return "", str(e)
        except Exception as e:
            print(f"An unexpected error occurred while calling the OpenAI API: {e}")
            return "", str(e)

class Gemini(LLMInterface):
    """
    Concrete implementation of LLMInterface using Google's Gemini 2.5 Pro model.

    Attributes:
        model (genai.GenerativeModel): Instance for interacting with the Gemini API.
    """

    def __init__(self):
        """
        Initializes the GeminiLLM class by configuring the API key and creating an 
        instance of the Gemini model.
        """
        super().__init__()
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables.")
            genai.configure(api_key=api_key)
            # Using a powerful and recent model. You can change this to other available models.
            self.model = genai.GenerativeModel('gemini-2.5-pro')
        except Exception as e:
            print(f"Error during Gemini initialization: {e}")
            self.model = None
        self.name = 'gemini'

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        """
        Sends the user prompt to the Gemini model and retrieves the solution.
        """
        if not self.model:
            return "Error: Model not initialized.", None
            
        try:
            # Add the request_options parameter to set a timeout
            # Timeout is in seconds. 600 seconds = 10 minutes.
            response = self.model.generate_content(
                user_prompt,
                request_options={"timeout": 600} 
            )
            solution_text = response.text
            return solution_text, response
        except Exception as e:
            print(f"An error occurred while calling the Gemini API: {e}")
            return f"Error: {e}", None

class Claude(LLMInterface):
    """
    Concrete implementation of LLMInterface using Anthropic's Claude models.
    """

    def __init__(self):
        """
        Initializes the Claude class by creating an instance of the Anthropic client.
        """
        super().__init__()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=api_key, timeout=600.0)
        self.name = 'claude'

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        """
        Sends the combined user prompt to Anthropic's model.

        Args:
            user_prompt (str): The complete prompt (system + problem).

        Returns:
            Tuple[str, Any]: The LLM's response and metadata.
        """
        try:
            # **FIX:** Removed the 'system' parameter.
            # The 'user_prompt' now contains the full concatenated prompt.
            completion = self.client.messages.create(
                model="claude-sonnet-4-20250514", 
                max_tokens=32000,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                thinking={
                    "type": "enabled",
                    "budget_tokens": 20000
                }
            )

            # --- THIS IS THE CORRECTED LOGIC ---
            # Instead of assuming content[0].text, we loop through all content blocks
            # and build the final text string from "text" blocks only.
            
            final_text = ""
            if hasattr(completion, 'content') and completion.content:
                for block in completion.content:
                    # Only add the text if it's a 'text' block
                    if hasattr(block, 'type') and block.type == 'text':
                        if hasattr(block, 'text'):
                            final_text += block.text
                    # Any other block type (like 'thinking') is safely ignored.
            
            return final_text, str(completion)
            # --- END OF CORRECTION ---

        except AnthropicAPITimeoutError as e:
            print(f"Anthropic API request timed out: {e}")
            return "", str(e)
        except Exception as e:
            print(f"An unexpected error occurred while calling the Anthropic API: {e}")
            return "", str(e)

class Claude_Opus(LLMInterface):
    """
    Concrete implementation of LLMInterface using Anthropic's Claude models.
    """

    def __init__(self):
        """
        Initializes the Claude class by creating an instance of the Anthropic client.
        """
        super().__init__()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=api_key, timeout=600.0)
        self.name = 'claude'

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        """
        Sends the combined user prompt to Anthropic's model.

        Args:
            user_prompt (str): The complete prompt (system + problem).

        Returns:
            Tuple[str, Any]: The LLM's response and metadata.
        """
        try:
            # **FIX:** Removed the 'system' parameter.
            # The 'user_prompt' now contains the full concatenated prompt.
            completion = self.client.messages.create(
                model="claude-opus-4-1-20250805", 
                max_tokens=32000,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                thinking={
                    "type": "enabled",
                    "budget_tokens": 20000
                }
            )

            # --- THIS IS THE CORRECTED LOGIC ---
            # Instead of assuming content[0].text, we loop through all content blocks
            # and build the final text string from "text" blocks only.
            
            final_text = ""
            if hasattr(completion, 'content') and completion.content:
                for block in completion.content:
                    # Only add the text if it's a 'text' block
                    if hasattr(block, 'type') and block.type == 'text':
                        if hasattr(block, 'text'):
                            final_text += block.text
                    # Any other block type (like 'thinking') is safely ignored.
            
            return final_text, str(completion)
            # --- END OF CORRECTION ---

        except AnthropicAPITimeoutError as e:
            print(f"Anthropic API request timed out: {e}")
            return "", str(e)
        except Exception as e:
            print(f"An unexpected error occurred while calling the Anthropic API: {e}")
            return "", str(e)

class Claude_Sonnet_4_5(LLMInterface):
    """
    Concrete implementation of LLMInterface using Anthropic's Claude models.
    """

    def __init__(self):
        """
        Initializes the Claude class by creating an instance of the Anthropic client.
        """
        super().__init__()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=api_key, timeout=600.0)
        self.name = 'claude'

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        """
        Sends the combined user prompt to Anthropic's model.

        Args:
            user_prompt (str): The complete prompt (system + problem).

        Returns:
            Tuple[str, Any]: The LLM's response and metadata.
        """
        try:
            # **FIX:** Removed the 'system' parameter.
            # The 'user_prompt' now contains the full concatenated prompt.
            completion = self.client.messages.create(
                model="claude-sonnet-4-5-20250929", 
                max_tokens=32000,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                thinking={
                    "type": "enabled",
                    "budget_tokens": 20000
                }
            )

            # --- THIS IS THE CORRECTED LOGIC ---
            # Instead of assuming content[0].text, we loop through all content blocks
            # and build the final text string from "text" blocks only.
            
            final_text = ""
            if hasattr(completion, 'content') and completion.content:
                for block in completion.content:
                    # Only add the text if it's a 'text' block
                    if hasattr(block, 'type') and block.type == 'text':
                        if hasattr(block, 'text'):
                            final_text += block.text
                    # Any other block type (like 'thinking') is safely ignored.
            
            return final_text, str(completion)
            # --- END OF CORRECTION ---

        except AnthropicAPITimeoutError as e:
            print(f"Anthropic API request timed out: {e}")
            return "", str(e)
        except Exception as e:
            print(f"An unexpected error occurred while calling the Anthropic API: {e}")
            return "", str(e)

class Grok(LLMInterface):
    """
    Concrete implementation of LLMInterface using xAI's Grok models.
    """

    def __init__(self):
        """
        Initializes the Grok class by creating an instance of the OpenAI client
        pointed at the Grok API endpoint.
        """
        super().__init__()
        api_key = os.getenv("XAI_API_KEY")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=1200.0
        )
        self.name = 'grok'

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        """
        Sends the combined user prompt to Grok's model.

        Args:
            user_prompt (str): The complete prompt (system + problem).

        Returns:
            Tuple[str, Any]: The LLM's response and metadata.
        """
        try:
            # Reverted to the simpler, single-message format
            completion = self.client.chat.completions.create(
                model="grok-4-0709", # Using a powerful Grok model
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return completion.choices[0].message.content, str(completion)
        except APITimeoutError as e:
            print(f"Grok (xAI) API request timed out: {e}")
            return "", str(e)
        except Exception as e:
            print(f"An unexpected error occurred while calling the Grok (xAI) API: {e}")
            return "", str(e)
    
class GPT_level(LLMInterface):
    """
    Concrete implementation of LLMInterface using OpenAI's GPT-4o model.
    """

    def __init__(self, level):
        """
        Initializes the ExampleLLM class by creating an instance of the OpenAI client.
        """
        super().__init__()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.name = 'gpt' + '-' + level
        self.reasoning_level = level

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        """
        Sends the user prompt to OpenAI's GPT-5 model and retrieves the solution.

        Args:
            user_prompt (str): The complete prompt including the initial context and problem statement.

        Returns:
            Tuple[str, dict]: The LLM's response and a dictionary with token usage.
        """
        try:
            completion = self.client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": user_prompt}],
                reasoning_effort=self.reasoning_level,
                timeout=1200.0
            )
            
            usage_data = None
            if hasattr(completion, 'usage'):
                usage_data = {
                    'prompt_tokens': completion.usage.prompt_tokens,
                    'completion_tokens': completion.usage.completion_tokens,
                    'total_tokens': completion.usage.total_tokens
                }
            
            return completion.choices[0].message.content, usage_data
        
        except APITimeoutError as e:
            print(f"OpenAI API request timed out: {e}")
            return "", None
        except Exception as e:
            print(f"An unexpected error occurred while calling the OpenAI API: {e}")
            return "", None

class Gemini3(LLMInterface):
    def __init__(self):
        """
        Initializes the OpenRouter class.
        
        Args:
            model_name (str): The specific model ID on OpenRouter (e.g., 'anthropic/claude-3-opus', 'meta-llama/llama-3-70b-instruct').
                              Defaults to a fast generic model.
        """
        super().__init__()
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        # OpenRouter is compatible with the OpenAI SDK, we just change the base_url
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name = "google/gemini-3-pro-preview"
        self.name = 'gemini3'

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        """
        Sends the user prompt to OpenRouter and retrieves the solution.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=1200.0,
                extra_body={"reasoning": {"enabled": True}}
            )
            return completion.choices[0].message.content, str(completion)
        except APITimeoutError as e:
            print(f"OpenRouter API request timed out: {e}")
            return "", str(e)
        except Exception as e:
            print(f"An unexpected error occurred while calling the OpenRouter API: {e}")
            return "", str(e)

class GPT_5_1(LLMInterface):
    """
    Concrete implementation of LLMInterface using OpenAI's GPT-4o model.
    """

    def __init__(self):
        """
        Initializes the ExampleLLM class by creating an instance of the OpenAI client.
        """
        super().__init__()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.name = 'gpt'

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        """
        Sends the user prompt to OpenAI's GPT-5 model and retrieves the solution.

        Args:
            user_prompt (str): The complete prompt including the initial context and problem statement.

        Returns:
            Tuple[str, Any]: The LLM's response and metadata about the completion.
        """
        try:
            completion = self.client.chat.completions.create(
                model="gpt-5.1",
                messages=[{"role": "user", "content": user_prompt}],
                reasoning_effort='high',
                timeout=1200.0
            )
            
            usage_data = None
            if hasattr(completion, 'usage'):
                usage_data = {
                    'prompt_tokens': completion.usage.prompt_tokens,
                    'completion_tokens': completion.usage.completion_tokens,
                    'total_tokens': completion.usage.total_tokens
                }
            
            return completion.choices[0].message.content, usage_data
        
        except APITimeoutError as e:
            print(f"OpenAI API request timed out: {e}")
            return "", None
        except Exception as e:
            print(f"An unexpected error occurred while calling the OpenAI API: {e}")
            return "", None

class Gemini3_level(LLMInterface):
    def __init__(self, thinking_level): # <--- Added timeout parameter
        """
        Initializes the official Google Gen AI SDK (v2).
        
        Args:
            thinking_level (str): "low" (faster) or "high" (deeper reasoning).
            timeout_seconds (int): Maximum time in seconds to wait for a response.
                                   Defaults to 600s (10 minutes).
        """
        timeout_seconds = 1200
        super().__init__()
        self.api_key = os.getenv("GOOGLE_API_KEY") 
        
        # --- NEW: Configure Timeout ---
        # The SDK expects timeout in MILLISECONDS.
        if timeout_seconds:
            http_options = types.HttpOptions(timeout=timeout_seconds * 1000)
        else:
            http_options = None

        # Pass http_options to the Client
        self.client = genai.Client(
            api_key=self.api_key, 
            http_options=http_options
        )
        # -----------------------------

        self.model_name = "gemini-3-pro-preview"
        self.thinking_level = thinking_level
        self.name = 'gemini3' + self.thinking_level

    def call_llm(self, user_prompt: str) -> Tuple[str, Dict[str, int]]:
        """
        Generates content using the native Google SDK with thinking parameters.
        """
        try:
            think_config = types.ThinkingConfig(thinking_level=self.thinking_level)
            
            gen_config = types.GenerateContentConfig(
                thinking_config=think_config
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=gen_config
            )

            text_content = response.text

            usage_meta = response.usage_metadata
            usage_data = {
                'prompt_tokens': usage_meta.prompt_token_count if usage_meta else 0,
                'completion_tokens': usage_meta.candidates_token_count if usage_meta else 0,
                'total_tokens': usage_meta.total_token_count if usage_meta else 0
            }

            return text_content, usage_data

        except Exception as e:
            # This will now catch the timeout error specifically
            print(f"Gemini Native SDK Error: {e}")
            return None, None

class Claude_Opus_4_5(LLMInterface):
    """
    Concrete implementation of LLMInterface using Anthropic's Claude models.
    """

    def __init__(self):
        """
        Initializes the Claude class by creating an instance of the Anthropic client.
        """
        super().__init__()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=api_key, timeout=1200.0)
        self.name = 'claude-opus-4-5'

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        """
        Sends the combined user prompt to Anthropic's model.

        Args:
            user_prompt (str): The complete prompt (system + problem).

        Returns:
            Tuple[str, Any]: The LLM's response and metadata.
        """
        try:
            # **FIX:** Removed the 'system' parameter.
            # The 'user_prompt' now contains the full concatenated prompt.
            completion = self.client.messages.create(
                model="claude-opus-4-5-20251101", 
                max_tokens=32000,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                thinking={
                    "type": "enabled",
                    "budget_tokens": 20000
                }
            )

            # --- THIS IS THE CORRECTED LOGIC ---
            # Instead of assuming content[0].text, we loop through all content blocks
            # and build the final text string from "text" blocks only.
            
            final_text = ""
            if hasattr(completion, 'content') and completion.content:
                for block in completion.content:
                    # Only add the text if it's a 'text' block
                    if hasattr(block, 'type') and block.type == 'text':
                        if hasattr(block, 'text'):
                            final_text += block.text
                    # Any other block type (like 'thinking') is safely ignored.
            
            return final_text, str(completion)
            # --- END OF CORRECTION ---

        except AnthropicAPITimeoutError as e:
            print(f"Anthropic API request timed out: {e}")
            return "", str(e)
        except Exception as e:
            print(f"An unexpected error occurred while calling the Anthropic API: {e}")
            return "", str(e)