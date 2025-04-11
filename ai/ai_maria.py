from datetime import datetime
import json
import requests
import math

class MariaAI:
    def __init__(self, memory: dict, model_url: str):
        self.memory = memory or {"personalidade": "Indefinida", "valores": [], "gatilhos_emocionais": [], 
                                 "reflexoes": [], "planos": [], "elogios": []}
        self.model_url = model_url
        self.MAX_TOKENS_PER_BATCH = 2000  # Limite seguro por lote

    def generate_initial_memory(self, conversation_turns: list):
        # Dividir turnos em lotes
        batches = self._split_into_batches(conversation_turns)
        print(f"📝 Gerando memória inicial para Maria em {len(batches)} lotes.")

        accumulated_profile = {
            "personalidade": "Indefinida",
            "valores": [],
            "gatilhos_emocionais": [],
            "reflexoes": [],
            "planos": [],
            "elogios": []
        }

        previous_feedback = ""
        for i, batch in enumerate(batches):
            conversation_text = self.format_conversation(batch)
            prompt = f"""
            Analisa APENAS esta parte da conversa entre Rui e Maria (lote {i+1}/{len(batches)}) e atualiza o perfil psicológico da Maria.
            Inclui:
            - personalidade (ex.: empática, extrovertida)
            - valores (ex.: compreensão, diálogo)
            - gatilhos emocionais (ex.: rejeição, ignorada)
            Retorna SOMENTE o JSON com essas chaves no nível raiz, sem texto adicional ou aninhamento.
            Exemplo: {{"personalidade": "empática", "valores": ["compreensão"], "gatilhos_emocionais": ["rejeição"]}}

            Feedback anterior (para contexto, se aplicável):
            {previous_feedback}

            Conversa:
            {conversation_text}
            """
            print(f"📋 Prompt para lote {i+1}: {prompt}s")
            print(f"📏 Tamanho estimado do prompt: ~{len(prompt.split()) // 0.75} tokens")
            print(f"📜 Conversa enviada (lote {i+1}): {conversation_text}")
            response = self.ask_model(prompt)
            profile_text = response.get("choices", [{}])[0].get("text", "{}").strip()

            # Extrair JSON
            import re
            json_match = re.search(r'\{.*\}', profile_text, re.DOTALL)
            if json_match:
                profile_json = json_match.group(0)
                try:
                    profile_data = json.loads(profile_json)
                    accumulated_profile["personalidade"] = profile_data.get("personalidade", accumulated_profile["personalidade"])
                    accumulated_profile["valores"] = list(set(accumulated_profile["valores"] + profile_data.get("valores", [])))
                    accumulated_profile["gatilhos_emocionais"] = list(set(accumulated_profile["gatilhos_emocionais"] + profile_data.get("gatilhos_emocionais", [])))
                    previous_feedback = profile_json
                except json.JSONDecodeError:
                    print(f"Erro ao parsear JSON do perfil de Maria (lote {i+1}): {profile_json}")
            else:
                print(f"Nenhum JSON válido encontrado para Maria (lote {i+1}): {profile_text}")

        # Finalizar memória
        self.memory = accumulated_profile
        self.memory.setdefault("reflexoes", [])
        self.memory.setdefault("planos", [])
        self.memory.setdefault("elogios", [])
        print(f"Memória inicial gerada para Maria: {self.memory}")

    def analyze(self, conversation_turns: list) -> dict:
        # Dividir turnos em lotes
        batches = self._split_into_batches(conversation_turns)
        print(f"📝 Analisando conversa para Maria em {len(batches)} lotes.")

        accumulated_feedback = {
            "reflexoes": [],
            "planos": [],
            "elogios": [],
            "valores": self.memory.get("valores", []),
            "gatilhos_emocionais": self.memory.get("gatilhos_emocionais", [])
        }
        previous_feedback = ""

        for i, batch in enumerate(batches):
            conversation_text = self.format_conversation(batch)
            prompt = self.generate_prompt(conversation_text, previous_feedback, batch_number=i+1, total_batches=len(batches))
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

            accumulated_feedback["reflexoes"].extend(reflexoes)
            accumulated_feedback["planos"].extend(planos)
            accumulated_feedback["elogios"].extend(elogios)
            
            if "compreensão" in feedback_text.lower() and "compreensão" not in accumulated_feedback["valores"]:
                accumulated_feedback["valores"].append("compreensão")
            if "rejeição" in feedback_text.lower() and "rejeição" not in accumulated_feedback["gatilhos_emocionais"]:
                accumulated_feedback["gatilhos_emocionais"].append("rejeição")
            
            previous_feedback = feedback_text
            print(f"Feedback parcial Maria (lote {i+1}): {feedback_text}")

        # Atualizar memória
        self.memory["reflexoes"].extend(accumulated_feedback["reflexoes"])
        self.memory["planos"].extend(accumulated_feedback["planos"])
        self.memory["elogios"].extend(accumulated_feedback["elogios"])
        self.memory["valores"] = list(set(self.memory["valores"] + accumulated_feedback["valores"]))
        self.memory["gatilhos_emocionais"] = list(set(self.memory["gatilhos_emocionais"] + accumulated_feedback["gatilhos_emocionais"]))
        
        return accumulated_feedback

    def format_conversation(self, turns: list) -> str:
        formatted = ""
        for turn in turns:
            for msg in turn:
                sender = msg.get("sender_name", "Desconhecido")
                content = msg.get("content", "")
                formatted += f"{sender}: {content}\n"
        return formatted.strip()

    def generate_prompt(self, conversation_text: str, previous_feedback: str, batch_number: int, total_batches: int) -> str:
        personalidade = self.memory.get("personalidade", "Indefinida")
        valores = ", ".join(self.memory.get("valores", []))
        gatilhos = ", ".join(self.memory.get("gatilhos_emocionais", []))
        return f"""
        Tu és a Maria. A tua personalidade é {personalidade}. Valorizas {valores}. Os teus gatilhos emocionais incluem {gatilhos}.
        A partir desta parte da conversa (lote {batch_number}/{total_batches}), gera um relatório emocional focando-te na Maria.
        Reflexões emocionais, planos futuros e elogios. Fala na primeira pessoa.
        Considera o feedback anterior para manter consistência:
        
        Feedback anterior:
        {previous_feedback}
        
        Conversa:
        {conversation_text}
        Relatório emocional:
        """

    def _split_into_batches(self, turns: list) -> list:
        """Divide os turnos em lotes com base no limite de tokens."""
        batches = []
        current_batch = []
        current_tokens = 0
        words_per_token = 0.75

        for turn in turns:
            turn_text = self.format_conversation([turn])
            word_count = len(turn_text.split())
            token_count = math.ceil(word_count / words_per_token)

            if current_tokens + token_count > self.MAX_TOKENS_PER_BATCH:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [turn]
                current_tokens = token_count
            else:
                current_batch.append(turn)
                current_tokens += token_count

        if current_batch:
            batches.append(current_batch)

        return batches

    def ask_model(self, prompt: str) -> dict:
        headers = {'Content-Type': 'application/json'}
        data = {'model': 'openhermes-2.5-mistral-7b', 'prompt': prompt, 'max_tokens': 800}
        try:
            response = requests.post(f"{self.model_url}/v1/completions", headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Erro ao chamar a API do modelo: {str(e)}")