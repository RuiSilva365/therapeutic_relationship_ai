from datetime import datetime
import json
import requests

class MariaAI:
    def __init__(self, memory: dict, model_url: str):
        self.memory = memory or {"personalidade": "Indefinida", "valores": [], "gatilhos_emocionais": [], 
                                 "reflexoes": [], "planos": [], "elogios": []}
        self.model_url = model_url

    def generate_initial_memory(self, conversation_turns: list):
        conversation_text = self.format_conversation(conversation_turns)
        prompt = f"""
        Analisa APENAS esta conversa entre Rui e Maria e cria um perfil psicológico inicial para a Maria.
        Inclui:
        - personalidade (ex.: empática, extrovertida)
        - valores (ex.: compreensão, diálogo)
        - gatilhos emocionais (ex.: rejeição, ignorada)
        Retorna SOMENTE o JSON com essas chaves no nível raiz, sem texto adicional ou aninhamento.
        Exemplo: {{"personalidade": "empática", "valores": ["compreensão"], "gatilhos_emocionais": ["rejeição"]}}
        
        Conversa:
        {conversation_text}
        """
        response = self.ask_model(prompt)
        profile_text = response.get("choices", [{}])[0].get("text", "{}").strip()
        print(f"Texto bruto retornado para Maria: {profile_text}")  # Depuração
        
        # Remover texto extra e extrair JSON
        import re
        json_match = re.search(r'\{.*\}', profile_text, re.DOTALL)
        if json_match:
            profile_json = json_match.group(0)
            try:
                profile_data = json.loads(profile_json)
                self.memory = profile_data
                self.memory.setdefault("reflexoes", [])
                self.memory.setdefault("planos", [])
                self.memory.setdefault("elogios", [])
            except json.JSONDecodeError:
                print(f"Erro ao parsear JSON do perfil de Maria: {profile_json}")
                self.memory = {"personalidade": "Indefinida", "valores": [], "gatilhos_emocionais": [],
                            "reflexoes": [], "planos": [], "elogios": []}
        else:
            print(f"Nenhum JSON válido encontrado para Maria: {profile_text}")
            self.memory = {"personalidade": "Indefinida", "valores": [], "gatilhos_emocionais": [],
                        "reflexoes": [], "planos": [], "elogios": []}
        print(f"Memória inicial gerada para Maria: {self.memory}")
        
    
    def analyze(self, conversation_turns: list) -> dict:
        conversation_text = self.format_conversation(conversation_turns)
        prompt = self.generate_prompt(conversation_text)
        feedback = self.ask_model(prompt)
        
        feedback_text = feedback.get("choices", [{}])[0].get("text", "")
        reflexoes, planos, elogios = [], [], []
        for line in feedback_text.split("\n"):
            line = line.strip()
            if "Reflexões emocionais" in line or "sinto" in line.lower():
                reflexoes.append({"data": datetime.now().strftime("%Y-%m-%d"), "texto": line})
            elif "Planos futuros" in line or "gostava de" in line.lower():
                planos.append({"data": datetime.now().strftime("%Y-%m-%d"), "texto": line})
            elif "Gestos de carinho" in line or "disse" in line.lower():
                elogios.append({"data": datetime.now().strftime("%Y-%m-%d"), "texto": line})
        
        self.memory["reflexoes"].extend(reflexoes)
        self.memory["planos"].extend(planos)
        self.memory["elogios"].extend(elogios)
        
        if "compreensão" in feedback_text.lower() and "compreensão" not in self.memory.get("valores", []):
            self.memory.setdefault("valores", []).append("compreensão")
        if "rejeição" in feedback_text.lower() and "rejeição" not in self.memory.get("gatilhos_emocionais", []):
            self.memory.setdefault("gatilhos_emocionais", []).append("rejeição")
        
        return feedback

    def format_conversation(self, turns: list) -> str:
        formatted = ""
        for turn in turns:
            for msg in turn:
                sender = msg.get("sender_name", "Desconhecido")
                content = msg.get("content", "")
                formatted += f"{sender}: {content}\n"
        return formatted.strip()

    def generate_prompt(self, conversation_text: str) -> str:
        personalidade = self.memory.get("personalidade", "Indefinida")
        valores = ", ".join(self.memory.get("valores", []))
        gatilhos = ", ".join(self.memory.get("gatilhos_emocionais", []))
        return f"""
        Tu és a Maria. A tua personalidade é {personalidade}. Valorizas {valores}. Os teus gatilhos emocionais incluem {gatilhos}.
        A partir desta conversa, gera um relatório emocional focando-te na Maria.
        Reflexões emocionais, planos futuros e elogios. Fala na primeira pessoa.
        
        Conversa:
        {conversation_text}
        Relatório emocional:
        """

    def ask_model(self, prompt: str) -> dict:
        headers = {'Content-Type': 'application/json'}
        data = {'model': 'openhermes-2.5-mistral-7b', 'prompt': prompt, 'max_tokens': 800}
        response = requests.post(f"{self.model_url}/v1/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Erro ao chamar a API do modelo: {response.status_code} - {response.text}")