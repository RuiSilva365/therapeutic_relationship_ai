import json
import requests  # type: ignore

class MariaAI:
    def __init__(self, memory: dict, model_url: str):
        """
        Inicializa a classe MariaAI com memória, URL do modelo e chave de API.
        :param memory: Dicionário que armazena a memória da IA.
        :param model_url: URL da API do modelo.
        """
        self.memory = memory or {"insights": [], "frustracoes": [], "afetos": []}
        self.model_url = model_url  # A URL agora deve apontar para o endpoint correto

    def analyze(self, conversation_turns: list) -> dict:
        """
        Analisa as conversas e gera um feedback emocional focado na Maria.
        :param conversation_turns: Lista de turnos de conversa a serem analisados.
        :return: Dicionário contendo o feedback gerado.
        """
        conversation_text = self.format_conversation(conversation_turns)
        prompt = self.generate_prompt(conversation_text)
        feedback = self.ask_model(prompt)
        return feedback

    def format_conversation(self, turns: list) -> str:
        """
        Formata os turnos de conversa em uma string para envio ao modelo.
        :param turns: Lista de turnos de conversa.
        :return: String formatada representando a conversa.
        """
        formatted = ""
        for turn in turns:
            for msg in turn:
                sender = msg.get("sender_name", "Desconhecido")
                content = msg.get("content", "")
                formatted += f"{sender}: {content}\n"
        return formatted.strip()

    def generate_prompt(self, conversation_text: str) -> str:
        """
        Gera o prompt a ser enviado ao modelo com base na conversa.
        :param conversation_text: Texto da conversa a ser analisada.
        :return: Prompt formatado para o modelo.
        """
        return f"""
        Tu és a Maria, tens de estar na pele dela. A partir desta conversa, gera um relatório emocional focando-te na Maria.
        Insights emocionais da Maria, frustrações e gestos de carinho.Não te esqueças de incluir a tua percepção sobre a conversa(como te fez sentir e o que pensas sobre isso). Tens de ser honesto e direto e de falar sempre na primeira pessoa(afinal de contas, tu és a Maria)
        Conversa:
        {conversation_text}
        Relatório emocional:
        """

    def ask_model(self, prompt: str) -> dict:
        """
        Envia o prompt para a API do modelo e retorna a resposta.
        :param prompt: Texto a ser enviado ao modelo.
        :return: Resposta do modelo em formato de dicionário.
        """
        headers = {
            'Content-Type': 'application/json',
        }
        data = {
            'model': 'openhermes-2.5-mistral-7b',  # Nome do modelo
            'prompt': prompt,
            'max_tokens': 800
        }

        # URL corrigida para o endpoint de completions
        model_url = f"{self.model_url}/v1/completions"  # Ajuste a URL conforme necessário

        response = requests.post(model_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Erro ao chamar a API do modelo: {response.status_code} - {response.text}")
