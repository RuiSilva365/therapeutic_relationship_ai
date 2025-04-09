import requests  # type: ignore
import json

class BaseAI:
    def __init__(self, memory: dict, model_url: str):
        """
        Inicializa a base da IA com a memória e URL completa da API do modelo.
        :param memory: Dicionário que armazena a memória da IA.
        :param model_url: URL completa da API (ex: "http://192.168.56.1:1234/v1/completions").
        """
        self.memory = memory
        self.model_url= model_url  # A URL agora deve apontar para o endpoint correto

    def _call_model_api(self, prompt: str) -> dict:
        """
        Faz a chamada à API do modelo com o prompt fornecido e retorna a resposta.
        :param prompt: Texto enviado ao modelo.
        :return: Resposta do modelo como dicionário.
        """
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            'model': 'openhermes-2.5-mistral-7b',
            'prompt': prompt,
            'max_tokens': 500
        }
        self.model_url = f"{self.model_url}/v1/completions"  # Ajuste a URL conforme necessário
        response = requests.post(self.model_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Erro ao chamar a API do modelo: {response.status_code} - {response.text}")

    def save_to_memory(self, key: str, value: dict):
        self.memory[key] = value

    def get_from_memory(self, key: str) -> dict:
        return self.memory.get(key, {})

    def update_memory(self, key: str, data: dict):
        if key in self.memory:
            self.memory[key].update(data)
        else:
            self.memory[key] = data
