import requests
import json
import re
from typing import Dict, Any
from datetime import datetime

class BaseAI:
    MEMORY_SCHEMA = {
        "personality": {"traits": [], "description": ""},
        "core_values": [],
        "emotional_patterns": [],
        "relational_dynamics": {"strengths": [], "challenges": [], "patterns": []},
        "struggles": [],
        "memories": [],
        "reflexoes": [],
        "planos": [],
        "elogios": [],
        "relationship_metrics": [],
        "insights": []
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
            elif isinstance(default_value, list) and key == "core_values":
                # Ensure core_values contains valid dictionaries
                valid_entries = []
                for item in self.memory[key]:
                    if isinstance(item, dict) and "value" in item:
                        valid_entries.append(item)
                    else:
                        print(f"🗑️ Entrada inválida descartada em core_values: {item}")
                self.memory[key] = valid_entries
        self.memory = {k: self.memory[k] for k in self.MEMORY_SCHEMA}
        print("✅ Memória validada conforme o esquema.")

    def update_memory(self, data: dict):
        for key, value in data.items():
            if key in self.MEMORY_SCHEMA:
                if isinstance(self.MEMORY_SCHEMA[key], list):
                    if key == "core_values":
                        # Validate core_values entries
                        valid_values = []
                        for item in value:
                            if isinstance(item, dict) and "value" in item:
                                valid_values.append(item)
                            elif isinstance(item, str):
                                print(f"⚠️ Convertendo string em core_values: {item}")
                                valid_values.append({
                                    "value": item,
                                    "manifestation": "",
                                    "evidence": ""
                                })
                            else:
                                print(f"🗑️ Ignorando entrada inválida em core_values: {item}")
                        self.memory[key].extend(valid_values)
                    else:
                        self.memory[key].extend(value)
                elif isinstance(self.MEMORY_SCHEMA[key], dict):
                    self.memory[key].update(value)
                else:
                    self.memory[key] = value
        self.validate_memory()

    @staticmethod
    def fix_encoding(text: str) -> str:
        try:
            return text.encode('utf-8').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text.replace('Ã©', 'é').replace('Ã£', 'ã').replace('ð', '')

    def _clean_json(self, text: str) -> str:
        if not text or text.strip() in ["", "{}"]:
            return json.dumps(self.MEMORY_SCHEMA, ensure_ascii=False)
        text = text.strip()
        print(f"🛠️ Texto bruto antes de limpeza: {text[:200]}...")
        try:
            data = json.loads(text)
            print("✅ JSON válido sem limpeza")
            # Fix core_values if it's a list of strings
            if "core_values" in data and isinstance(data["core_values"], list):
                valid_core_values = []
                for item in data["core_values"]:
                    if isinstance(item, dict) and "value" in item:
                        valid_core_values.append(item)
                    elif isinstance(item, str):
                        print(f"⚠️ Convertendo core_values string: {item}")
                        valid_core_values.append({
                            "value": item,
                            "manifestation": "",
                            "evidence": ""
                        })
                data["core_values"] = valid_core_values
            return json.dumps(data, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        # Remove markdown and comments
        text = re.sub(r'^```(?:json)?\n|```$', '', text, flags=re.MULTILINE)
        text = re.sub(r'//.*?\n|/\*.*?\*/', '', text, flags=re.DOTALL)
        # Fix single quotes
        text = re.sub(r"'([^']*)'", r'"\1"', text)
        # Fix unquoted keys
        text = re.sub(r'([{,]\s*)([^"{}\[\],\s:]+?)(:)', r'\1"\2"\3', text)
        # Fix unquoted values
        text = re.sub(r':\s*([^,\]\[}\s"][^,\]\[}]*)(?=[,\]\}])', r': "\1"', text)
        # Fix stray commas
        text = re.sub(r'([{[])\s*,(\s*")', r'\1\2', text)
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        # Fix missing commas
        text = re.sub(r'(\}\s*\{)', r'\},\1', text)
        # Fix incomplete structures
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        text += '}' * max(0, open_braces)
        text += ']' * max(0, open_brackets)
        try:
            data = json.loads(text)
            # Fix core_values
            if "core_values" in data and isinstance(data["core_values"], list):
                valid_core_values = []
                for item in data["core_values"]:
                    if isinstance(item, dict) and "value" in item:
                        valid_core_values.append(item)
                    elif isinstance(item, str):
                        print(f"⚠️ Convertendo core_values string: {item}")
                        valid_core_values.append({
                            "value": item,
                            "manifestation": "",
                            "evidence": ""
                        })
                data["core_values"] = valid_core_values
            # Normalize dates
            for key in ['reflexoes', 'planos', 'elogios', 'memories']:
                if key in data:
                    for item in data[key]:
                        date_key = 'data' if 'data' in item else 'date'
                        if date_key in item:
                            try:
                                datetime.strptime(item[date_key], "%Y-%m-%d")
                            except ValueError:
                                item[date_key] = "2025-04-12"
            text = json.dumps(data, ensure_ascii=False)
            print("✅ JSON válido após limpeza")
            return text
        except json.JSONDecodeError as e:
            print(f"⚠️ Falha na limpeza JSON: {str(e)}")
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    print("✅ JSON válido extraído via regex")
                    return json.dumps(data, ensure_ascii=False)
                except json.JSONDecodeError:
                    print(f"❌ Falha ao corrigir JSON extraído")
            print("🔙 Retornando esquema padrão como fallback")
            return json.dumps(self.MEMORY_SCHEMA, ensure_ascii=False)

    def _call_model_api(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.6) -> Dict[str, Any]:
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Accept-Charset': 'utf-8'
        }
        data = {
            'model': 'llama-3.2-1b-instruct',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        try:
            print(f"📤 Enviando request para {self.model_url}/v1/chat/completions")
            response = requests.post(f"{self.model_url}/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            response.encoding = 'utf-8'
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]
            print(f"📥 Resposta crua da API: {content[:200]}...")
            return {"choices": [{"text": content}]}
        except requests.RequestException as e:
            print(f"❌ Erro na API: {str(e)}")
            return {"choices": [{"text": json.dumps(self.MEMORY_SCHEMA, ensure_ascii=False)}]}