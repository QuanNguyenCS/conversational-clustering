import os
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

from .base_agent import BaseLLMAgent

def parse_json_from_text(text: str) -> Dict:
    """
    Robustly extract and parse a JSON object from text, 
    even if it is wrapped in markdown code blocks or has extra text.
    """
    text = text.strip()
    
    # Remove markdown code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
        
    # Extract the outermost JSON object
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx + 1]
        
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Return empty constraints if parsing fails
        print(f"Error parsing JSON from agent output: {e}\nRaw output was: {text}")
        return {"must_link": [], "cannot_link": []}


class OpenAIAgent(BaseLLMAgent):
    """LLM agent using OpenAI's API."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o", verbose: bool = False):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY environment variable.")
        
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.verbose = verbose

    def generate_constraints(self, history: List[Dict[str, str]], sampled_data: List[str]) -> Dict:
        # Formulate a prompt showing index and contents of the sampled data
        formatted_data = "\n".join([f"[{i}] {text}" for i, text in enumerate(sampled_data)])
        
        system_prompt = (
            "You are a helpful assistant in the Conversational Constrained Clustering (C3) system.\n"
            "Below is a sampled dataset (containing index [i] and text content):\n"
            f"{formatted_data}\n\n"
            "Based on the clustering requirements from the chat history, analyze which pairs of documents MUST belong to the same group (must-link) "
            "or CANNOT belong to the same group (cannot-link).\n"
            "Please return a single JSON object with the format:\n"
            "{\n"
            "  \"must_link\": [[i, j], ...],\n"
            "  \"cannot_link\": [[i, j], ...]\n"
            "}\n"
            "Where i, j are numerical indices of the documents in the sampled data above.\n"
            "NOTE: Do not explain anything else, only return valid JSON."
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history. Normalize roles to openai (system, user, assistant)
        for msg in history:
            role = msg["role"]
            if role not in ("system", "user", "assistant"):
                role = "user" if role == "human" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
            
        if self.verbose:
            print("\n" + "="*40 + " OPENAI REQUEST (generate_constraints) " + "="*40)
            print(f"Model: {self.model_name}")
            print("Messages:")
            for msg in messages:
                print(f"  Role: {msg.get('role')}")
                print(f"  Content:\n{msg.get('content')}")
                print("-" * 40)
            print("="*100)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            
            if self.verbose:
                print("\n" + "="*40 + " OPENAI RAW RESPONSE " + "="*40)
                print(content)
                print("="*100)
                
            return parse_json_from_text(content)
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            return {"must_link": [], "cannot_link": []}

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if self.verbose:
            print("\n" + "="*40 + " OPENAI REQUEST (generate_text) " + "="*40)
            print(f"Model: {self.model_name}")
            print("Messages:")
            for msg in messages:
                print(f"  Role: {msg.get('role')}")
                print(f"  Content:\n{msg.get('content')}")
                print("-" * 40)
            print("="*100)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            content = response.choices[0].message.content.strip()
            
            if self.verbose:
                print("\n" + "="*40 + " OPENAI RAW RESPONSE " + "="*40)
                print(content)
                print("="*100)
                
            return content
        except Exception as e:
            print(f"OpenAI raw generate_text failed: {e}")
            return ""


class GeminiAgent(BaseLLMAgent):
    """LLM agent using Google's Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-pro", verbose: bool = False):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY environment variable.")
        
        # pyrefly: ignore [missing-import]
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.genai = genai
        self.verbose = verbose

    def generate_constraints(self, history: List[Dict[str, str]], sampled_data: List[str]) -> Dict:
        formatted_data = "\n".join([f"[{i}] {text}" for i, text in enumerate(sampled_data)])
        
        system_prompt = (
            "You are a helpful assistant in the Conversational Constrained Clustering (C3) system.\n"
            "Below is a sampled dataset (containing index [i] and text content):\n"
            f"{formatted_data}\n\n"
            "Based on the clustering requirements from the chat history, analyze which pairs of documents MUST belong to the same group (must-link) "
            "or CANNOT belong to the same group (cannot-link).\n"
            "Please return a single JSON object with the structure:\n"
            "{\n"
            "  \"must_link\": [[i, j], ...],\n"
            "  \"cannot_link\": [[i, j], ...]\n"
            "}\n"
            "Do not explain anything else, only return valid JSON."
        )
        
        model = self.genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt
        )
        
        # Translate history to Gemini format (user, model)
        contents = []
        for msg in history:
            role = "user" if msg["role"] in ("user", "human") else "model"
            contents.append({"role": role, "parts": [msg["content"]]})
            
        # If history is empty, pass a default message to trigger response
        if not contents:
            contents.append({"role": "user", "parts": ["Extract clustering constraints from the sampled data based on defaults."]})
            
        if self.verbose:
            print("\n" + "="*40 + " GEMINI REQUEST (generate_constraints) " + "="*40)
            print(f"Model: {self.model_name}")
            print(f"System Prompt:\n{system_prompt}")
            print("-" * 40)
            print("Messages:")
            for msg in contents:
                print(f"  Role: {msg.get('role')}")
                print(f"  Content:\n{msg.get('parts')[0]}")
                print("-" * 40)
            print("="*100)

        try:
            response = model.generate_content(
                contents,
                generation_config={"response_mime_type": "application/json"}
            )
            content = response.text
            
            if self.verbose:
                print("\n" + "="*40 + " GEMINI RAW RESPONSE " + "="*40)
                print(content)
                print("="*100)
                
            return parse_json_from_text(content)
        except Exception as e:
            print(f"Gemini API call failed: {e}")
            return {"must_link": [], "cannot_link": []}

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        model = self.genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt
        )
        
        if self.verbose:
            print("\n" + "="*40 + " GEMINI REQUEST (generate_text) " + "="*40)
            print(f"Model: {self.model_name}")
            print(f"System Prompt:\n{system_prompt}")
            print("-" * 40)
            print(f"User Prompt:\n{user_prompt}")
            print("="*100)

        try:
            response = model.generate_content(user_prompt)
            content = response.text.strip()
            
            if self.verbose:
                print("\n" + "="*40 + " GEMINI RAW RESPONSE " + "="*40)
                print(content)
                print("="*100)
                
            return content
        except Exception as e:
            print(f"Gemini raw generate_text failed: {e}")
            return ""


class AnthropicAgent(BaseLLMAgent):
    """LLM agent using Anthropic's Claude API."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-3-5-sonnet-20240620", verbose: bool = False):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is missing. Set ANTHROPIC_API_KEY environment variable.")
            
        from anthropic import Anthropic
        self.client = Anthropic(api_key=self.api_key)
        self.model_name = model_name
        self.verbose = verbose

    def generate_constraints(self, history: List[Dict[str, str]], sampled_data: List[str]) -> Dict:
        formatted_data = "\n".join([f"[{i}] {text}" for i, text in enumerate(sampled_data)])
        
        system_prompt = (
            "You are a helpful assistant in the Conversational Constrained Clustering (C3) system.\n"
            "Below is a sampled dataset (containing index [i] and text content):\n"
            f"{formatted_data}\n\n"
            "Based on the clustering requirements from the chat history, analyze which pairs of documents MUST belong to the same group (must-link) "
            "or CANNOT belong to the same group (cannot-link).\n"
            "Please return a single JSON object with the structure:\n"
            "{\n"
            "  \"must_link\": [[i, j], ...],\n"
            "  \"cannot_link\": [[i, j], ...]\n"
            "}\n"
            "Do not explain anything else, only return valid JSON."
        )
        
        # Anthropic messages must alternate user/assistant and cannot be empty
        anthropic_messages = []
        for msg in history:
            role = "user" if msg["role"] in ("user", "human") else "assistant"
            anthropic_messages.append({"role": role, "content": msg["content"]})
            
        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": "Extract clustering constraints."})
            
        if self.verbose:
            print("\n" + "="*40 + " ANTHROPIC REQUEST (generate_constraints) " + "="*40)
            print(f"Model: {self.model_name}")
            print(f"System Prompt:\n{system_prompt}")
            print("-" * 40)
            print("Messages:")
            for msg in anthropic_messages:
                print(f"  Role: {msg.get('role')}")
                print(f"  Content:\n{msg.get('content')}")
                print("-" * 40)
            print("="*100)

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                system=system_prompt,
                messages=anthropic_messages
            )
            content = response.content[0].text
            
            if self.verbose:
                print("\n" + "="*40 + " ANTHROPIC RAW RESPONSE " + "="*40)
                print(content)
                print("="*100)
                
            return parse_json_from_text(content)
        except Exception as e:
            print(f"Anthropic API call failed: {e}")
            return {"must_link": [], "cannot_link": []}

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        
        if self.verbose:
            print("\n" + "="*40 + " ANTHROPIC REQUEST (generate_text) " + "="*40)
            print(f"Model: {self.model_name}")
            print(f"System Prompt:\n{system_prompt}")
            print("-" * 40)
            print("Messages:")
            for msg in messages:
                print(f"  Role: {msg.get('role')}")
                print(f"  Content:\n{msg.get('content')}")
                print("-" * 40)
            print("="*100)

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                system=system_prompt,
                messages=messages
            )
            content = response.content[0].text.strip()
            
            if self.verbose:
                print("\n" + "="*40 + " ANTHROPIC RAW RESPONSE " + "="*40)
                print(content)
                print("="*100)
                
            return content
        except Exception as e:
            print(f"Anthropic raw generate_text failed: {e}")
            return ""


class GitHubModelsAgent(BaseLLMAgent):
    """LLM agent using GitHub Models (OpenAI client with Azure endpoint)."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini", verbose: bool = False):
        self.api_key = api_key or os.getenv("GITHUB_TOKEN")
        if not self.api_key:
            raise ValueError("GitHub Token is missing. Set GITHUB_TOKEN environment variable.")
        
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=self.api_key
        )
        self.model_name = model_name
        self.verbose = verbose
        self.input_tokens = 0
        self.output_tokens = 0

    def reset_token_counters(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def generate_constraints(self, history: List[Dict[str, str]], sampled_data: List[str]) -> Dict:
        formatted_data = "\n".join([f"[{i}] {text}" for i, text in enumerate(sampled_data)])
        
        system_prompt = (
            "You are a helpful assistant in the Conversational Constrained Clustering (C3) system.\n"
            "Below is a sampled dataset (containing index [i] and text content):\n"
            f"{formatted_data}\n\n"
            "Based on the clustering requirements from the chat history, analyze which pairs of documents MUST belong to the same group (must-link) "
            "or CANNOT belong to the same group (cannot-link).\n"
            "Please return a single JSON object with the format:\n"
            "{\n"
            "  \"must_link\": [[i, j], ...],\n"
            "  \"cannot_link\": [[i, j], ...]\n"
            "}\n"
            "Where i, j are numerical indices of the documents in the sampled data above.\n"
            "NOTE: Do not explain anything else, only return valid JSON."
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            role = msg["role"]
            if role not in ("system", "user", "assistant"):
                role = "user" if role == "human" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
            
        if self.verbose:
            print("\n" + "="*40 + " GITHUB MODELS REQUEST (generate_constraints) " + "="*40)
            print(f"Model: {self.model_name}")
            print("Messages:")
            for msg in messages:
                print(f"  Role: {msg.get('role')}")
                print(f"  Content:\n{msg.get('content')}")
                print("-" * 40)
            print("="*100)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            content = response.choices[0].message.content
            
            if hasattr(response, "usage") and response.usage:
                self.input_tokens += getattr(response.usage, "prompt_tokens", 0)
                self.output_tokens += getattr(response.usage, "completion_tokens", 0)
            
            if self.verbose:
                print("\n" + "="*40 + " GITHUB MODELS RAW RESPONSE " + "="*40)
                print(content)
                print("="*100)
                
            return parse_json_from_text(content)
        except Exception as e:
            print(f"GitHub Models API call failed: {e}")
            return {"must_link": [], "cannot_link": []}

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if self.verbose:
            print("\n" + "="*40 + " GITHUB MODELS REQUEST (generate_text) " + "="*40)
            print(f"Model: {self.model_name}")
            print("Messages:")
            for msg in messages:
                print(f"  Role: {msg.get('role')}")
                print(f"  Content:\n{msg.get('content')}")
                print("-" * 40)
            print("="*100)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            content = response.choices[0].message.content.strip()
            
            if hasattr(response, "usage") and response.usage:
                self.input_tokens += getattr(response.usage, "prompt_tokens", 0)
                self.output_tokens += getattr(response.usage, "completion_tokens", 0)
            
            if self.verbose:
                print("\n" + "="*40 + " GITHUB MODELS RAW RESPONSE " + "="*40)
                print(content)
                print("="*100)
                
            return content
        except Exception as e:
            print(f"GitHub Models raw generate_text failed: {e}")
            return ""
