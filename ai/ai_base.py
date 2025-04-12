import requests
import json
import re

class BaseAI:
    def __init__(self, memory: dict, model_url: str):
        self.memory = memory
        self.model_url = model_url.rstrip('/')

    @staticmethod
    def fix_encoding(text: str) -> str:
        # Try different encoding combinations
        encodings = [
            ('latin1', 'utf-8'),  # UTF-8 interpreted as Latin-1
            ('utf-8', 'latin1'),  # Latin-1 interpreted as UTF-8
            ('cp1252', 'utf-8')   # Windows-1252 interpreted as UTF-8
        ]
    
        for src, dest in encodings:
            try:
                return text.encode(src).decode(dest)
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
    
        return text  # Return original if all fail
    
    def _clean_json(self, text: str) -> str:
        """Tenta limpar JSON malformado."""
        text = text.strip()
        # Remove ```json ou ```
        text = re.sub(r'^```(?:json)?\n|```$', '', text, flags=re.MULTILINE)
        # Corrige chaves sem aspas
        text = re.sub(r'([{,]\s*)(\w+)(:)', r'\1"\2"\3', text)
        # Remove vírgulas extras
        text = re.sub(r',\s*([\]}])', r'\1', text)
        # Substitui caracteres mal codificados
        
        text = self.fix_encoding(text)

        return text

    def _call_model_api(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.3) -> dict:
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
            # Força decodificação UTF-8
            response.encoding = 'utf-8'
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]
            print(f"📥 Resposta crua da API: {content[:200]}...")
            return {
                "choices": [{"text": content}]
            }
        except requests.RequestException as e:
            print(f"❌ Erro na API: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Detalhes do erro: {e.response.text}")
            return {"choices": [{"text": "{}"}]}
        
    def save_to_memory(self, key: str, value: dict):
        self.memory[key] = value

    def get_from_memory(self, key: str) -> dict:
        return self.memory.get(key, {})

    def update_memory(self, key: str, data: dict):
        if key in self.memory:
            self.memory[key].update(data)
        else:
            self.memory[key] = data