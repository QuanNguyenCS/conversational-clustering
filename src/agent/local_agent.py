from typing import List, Dict
import json
import requests
from .base_agent import BaseLLMAgent

class OllamaLocalAgent(BaseLLMAgent):
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "qwen2.5:7b", verbose: bool = False):
        self.base_url = f"{base_url}/api/chat"
        self.model_name = self._resolve_model(base_url, model_name)
        self.verbose = verbose

    def _resolve_model(self, base_url: str, model_name: str) -> str:
        try:
            res = requests.get(f"{base_url}/api/tags", timeout=5)
            if res.status_code == 200:
                models = [m["name"] for m in res.json().get("models", [])]
                # 1. Exact match
                if model_name in models:
                    return model_name
                # 2. Match tag prefix or sub-string (excluding embedding models)
                for m in models:
                    if "embed" in m.lower():
                        continue
                    if model_name.split(":")[0] in m:
                        print(f"[Ollama] Model '{model_name}' not found. Resolving to sibling model '{m}'.")
                        return m
                # 3. Match any compatible chat/generation model if qwen/llama/deepseek is requested
                if "qwen" in model_name.lower() or "llama" in model_name.lower() or "deepseek" in model_name.lower():
                    for m in models:
                        if "embed" in m.lower():
                            continue
                        if "qwen" in m.lower() or "llama" in m.lower() or "deepseek" in m.lower():
                            print(f"[Ollama] Model '{model_name}' not found. Resolving to compatible model '{m}'.")
                            return m
                # 4. Fallback to first non-embedding model
                for m in models:
                    if "embed" not in m.lower():
                        print(f"[Ollama] Model '{model_name}' not found. Falling back to first available model '{m}'.")
                        return m
        except Exception as e:
            print(f"[Ollama Warning] Failed to query tags to resolve model: {e}")
        return model_name

    def generate_constraints(self, history: List[Dict[str, str]], sampled_data: List[str]) -> Dict:
        system_prompt = (
            f"You are the C3 system middleware. Based on the sampled data:\n{sampled_data}\n"
            "Please return a single JSON object with structure: "
            '{"must_link": [[i, j]], "cannot_link": [[i, j]]}. Do not explain anything else.'
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "format": "json", # Force Ollama to return JSON format
            "stream": False
        }
        
        if self.verbose:
            print("\n" + "="*40 + " OLLAMA REQUEST (generate_constraints) " + "="*40)
            print(f"Model: {payload.get('model')}")
            print("Messages:")
            for msg in payload.get('messages', []):
                print(f"  Role: {msg.get('role')}")
                print(f"  Content:\n{msg.get('content')}")
                print("-" * 40)
            print("="*100)
            
        try:
            response = requests.post(self.base_url, json=payload)
            response.json_data = response.json()
            
            if self.verbose:
                print("\n" + "="*40 + " OLLAMA RAW RESPONSE " + "="*40)
                if isinstance(response.json_data, dict) and "message" in response.json_data:
                    print(response.json_data["message"].get("content"))
                else:
                    print(json.dumps(response.json_data, indent=2, ensure_ascii=False))
                print("="*100)
            
            if "error" in response.json_data:
                print(f"\n[Ollama Error] API returned error: {response.json_data['error']}")
                print("Make sure you have pulled the model using: ollama pull " + self.model_name)
                return {"must_link": [], "cannot_link": []}
                
            content = response.json_data["message"]["content"].strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            return json.loads(content)
        except Exception as e:
            print(f"\n[Ollama Connection Error] generate_constraints failed: {e}")
            return {"must_link": [], "cannot_link": []}

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }
        
        if self.verbose:
            print("\n" + "="*40 + " OLLAMA REQUEST (generate_text) " + "="*40)
            print(f"Model: {payload.get('model')}")
            print("Messages:")
            for msg in payload.get('messages', []):
                print(f"  Role: {msg.get('role')}")
                print(f"  Content:\n{msg.get('content')}")
                print("-" * 40)
            print("="*100)
            
        try:
            response = requests.post(self.base_url, json=payload)
            response_json = response.json()
            
            if self.verbose:
                print("\n" + "="*40 + " OLLAMA RAW RESPONSE " + "="*40)
                if isinstance(response_json, dict) and "message" in response_json:
                    print(response_json["message"].get("content"))
                else:
                    print(json.dumps(response_json, indent=2, ensure_ascii=False))
                print("="*100)
            
            if "error" in response_json:
                print(f"\n[Ollama Error] API returned error: {response_json['error']}")
                print("Make sure you have pulled the model using: ollama pull " + self.model_name)
                return f"Error: Ollama API returned error: {response_json['error']}"
                
            return response_json["message"]["content"]
        except Exception as e:
            print(f"\n[Ollama Connection Error] generate_text failed: {e}")
            return f"Error: {e}"
