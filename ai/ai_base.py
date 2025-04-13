import requests
import json
import re
from typing import Dict, Any
from datetime import datetime
import time

class BaseAI:
    MEMORY_SCHEMA = {
        "personality": {"traits": [], "description": ""},
        "core_values": [],
        "emotional_patterns": [],
        "relational_dynamics": {"strengths": [], "challenges": [], "patterns": []},
        "recent_reflections": []
    }

    def __init__(self, memory: dict, model_url: str):
        self.model_url = model_url.rstrip('/')
        self.memory = self.MEMORY_SCHEMA.copy()
        if memory:
            self.update_memory(memory)
        self.validate_memory()

    def validate_memory(self):
        for key, default_value in self.MEMORY_SCHEMA.items():
            if key not in self.memory:
                self.memory[key] = default_value
            elif isinstance(default_value, dict):
                for subkey, subvalue in default_value.items():
                    if subkey not in self.memory[key]:
                        self.memory[key][subkey] = subvalue
        self.memory = {k: self.memory[k] for k in self.MEMORY_SCHEMA}
        print("✅ Memória validada conforme o esquema.")

    def update_memory(self, data: dict):
        if not data or data == self.MEMORY_SCHEMA:
            print("⚠️ Dados vazios, pulando atualização de memória.")
            return
        for key, value in data.items():
            if key in self.MEMORY_SCHEMA:
                if key == "recent_reflections":
                    self.memory[key] = (self.memory[key] + value)[-5:]
                elif isinstance(self.MEMORY_SCHEMA[key], list):
                    self.memory[key] = value[:3]
                elif isinstance(self.MEMORY_SCHEMA[key], dict):
                    self.memory[key].update(value)
        self.validate_memory()

    def _should_update_profile(self, data: dict) -> bool:
        return len(self.memory.get("recent_reflections", [])) > 10

    @staticmethod
    def fix_encoding(text: str) -> str:
        if not text:
            return text
        try:
            return text.encode('utf-8').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            replacements = {
                'Ã©': 'é', 'Ã£': 'ã', 'Ã¢': 'â', 'Ã§': 'ç', 'Ã¡': 'á',
                'Ãª': 'ê', 'Ã­': 'í', 'Ã³': 'ó', 'Ãµ': 'õ', 'Ã´': 'ô',
                'Ãº': 'ú', 'Ã': '', 'ð': '', 'Â': ''
            }
            for wrong, right in replacements.items():
                text = text.replace(wrong, right)
            return text

    def _clean_json(self, text: str) -> str:
        if not text or text.strip() in ["", "{}"]:
            return json.dumps(self.MEMORY_SCHEMA, ensure_ascii=False)
        text = text.strip()
        try:
            data = json.loads(text)
            return json.dumps(data, ensure_ascii=False)
        except json.JSONDecodeError:
            text = re.sub(r'^```(?:json)?\n|```$', '', text, flags=re.MULTILINE)
            text = re.sub(r'//.*?\n|/\*.*?\*/', '', text, flags=re.DOTALL)
            text = re.sub(r"'([^']*)'", r'"\1"', text)
            text = re.sub(r'([{,]\s*)([^"{}\[\],\s:]+?)(:)', r'\1"\2"\3', text)
            text = re.sub(r':\s*([^,\]\[}\s"][^,\]\[}]*)(?=[,\]\}])', r': "\1"', text)
            text = re.sub(r'([{[])\s*,(\s*")', r'\1\2', text)
            text = re.sub(r',(\s*[}\]])', r'\1', text)
            open_braces = text.count('{') - text.count('}')
            open_brackets = text.count('[') - text.count(']')
            text += '}' * max(0, open_braces)
            text += ']' * max(0, open_brackets)
            try:
                data = json.loads(text)
                return json.dumps(data, ensure_ascii=False)
            except json.JSONDecodeError:
                return json.dumps(self.MEMORY_SCHEMA, ensure_ascii=False)

    def _call_model_api(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.6) -> Dict[str, Any]:
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Accept-Charset': 'utf-8'
        }
        data = {
            'model': 'hermes-3-llama-3.2-3b-q4_k_m',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': min(max_tokens, 4096),
            'temperature': temperature
        }
        retries = 3  # Reduced from 10
        for attempt in range(retries):
            try:
                prompt_tokens = len(prompt) // 4  # Conservative estimate
                print(f"📡 Enviando request (tentativa {attempt+1}): {json.dumps(data, ensure_ascii=False)[:200]}... (~{prompt_tokens} tokens)")
                response = requests.post(f"{self.model_url}/v1/chat/completions", headers=headers, json=data, timeout=60)
                response.raise_for_status()
                response.encoding = 'utf-8'
                response_json = response.json()
                content = response_json["choices"][0]["message"]["content"]
                print(f"📥 Resposta recebida: {content[:200]}...")
                return {"choices": [{"text": content}]}
            except requests.RequestException as e:
                print(f"❌ Erro na API (tentativa {attempt+1}/{retries}): {str(e)}")
                if hasattr(e.response, 'text'):
                    print(f"Detalhes do erro: {e.response.text}")
                    if "context length" in str(e.response.text).lower():
                        print("⚠️ Erro de limite de contexto detectado, abortando tentativas.")
                        return {"choices": [{"text": json.dumps(self.MEMORY_SCHEMA, ensure_ascii=False)}]}
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"❌ Falha após {retries} tentativas. Retornando schema padrão.")
                    return {"choices": [{"text": json.dumps(self.MEMORY_SCHEMA, ensure_ascii=False)}]}