from datetime import datetime
import json
import requests
import math

class RuiAI:
    def __init__(self, memory: dict, model_url: str):
        self.memory = memory or {"personalidade": "Indefinida", "valores": [], "gatilhos_emocionais": [], 
                                 "reflexoes": [], "planos": [], "elogios": []}
        self.model_url = model_url
        self.MAX_TOKENS_PER_BATCH = 1000  # Aumentado para mais contexto, ainda seguro abaixo de 4096

    def generate_initial_memory(self, conversation_turns: list):
        batches = self._split_into_batches(conversation_turns)
        print(f"📝 Gerando memória inicial para Rui em {len(batches)} lotes.")

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
            Analisa ESTA PARTE da conversa entre Rui e Maria (lote {i+1}/{len(batches)}) e atualiza o perfil psicológico do Rui com base apenas nas mensagens fornecidas.
            Inclui:
            - personalidade: descreve traços específicos (ex.: introspectivo, pragmático) com base em como Rui fala ou responde.
            - valores: identifica valores implícitos nas mensagens (ex.: conexão, honestidade) com exemplos breves.
            - gatilhos emocionais: detecta emoções fortes ou reações (ex.: stress, rejeição) e explica o que os causou.
            Retorna SOMENTE o JSON com essas chaves no nível raiz, sem texto adicional ou aninhamento.
            Exemplo: {{"personalidade": "introspectivo", "valores": ["conexão"], "gatilhos_emocionais": ["stress"]}}
            Se não houver informação suficiente, mantém os valores anteriores ou usa "Indefinido".

            Feedback anterior (para refinar, se aplicável):
            {previous_feedback}

            Conversa:
            {conversation_text}
            """
            print(f"📋 Prompt para lote {i+1}: {prompt}s")
            print(f"📏 Tamanho estimado do prompt: ~{len(prompt.split()) // 0.75} tokens")
            print(f"📜 Conversa enviada (lote {i+1}): {conversation_text[:500]}... (truncado para log)")

            try:
                response = self.ask_model(prompt)
                profile_text = response.get("choices", [{}])[0].get("text", "{}").strip()
                print(f"📤 Texto bruto retornado para Rui (lote {i+1}): {profile_text}")

                # Extrair JSON
                import re
                json_match = re.search(r'\{.*\}', profile_text, re.DOTALL)
                if json_match:
                    profile_json = json_match.group(0)
                    try:
                        profile_data = json.loads(profile_json)
                        # Atualizar personalidade (priorizar a mais recente, se definida)
                        if profile_data.get("personalidade") and profile_data["personalidade"] != "Indefinido":
                            accumulated_profile["personalidade"] = profile_data["personalidade"]
                        # Acumular valores e gatilhos, normalizando duplicatas
                        accumulated_profile["valores"] = list(set(accumulated_profile["valores"] + profile_data.get("valores", [])))
                        triggers = profile_data.get("gatilhos_emocionais", [])
                        # Normalizar "stress" e "estresse"
                        normalized_triggers = [t.lower().replace("estresse", "stress") for t in triggers]
                        accumulated_profile["gatilhos_emocionais"] = list(set(accumulated_profile["gatilhos_emocionais"] + normalized_triggers))
                        previous_feedback = json.dumps(profile_data, ensure_ascii=False)  # Passar como JSON limpo
                    except json.JSONDecodeError:
                        print(f"❌ Erro ao parsear JSON do perfil de Rui (lote {i+1}): {profile_json}")
                else:
                    print(f"❌ Nenhum JSON válido encontrado para Rui (lote {i+1}): {profile_text}")
            except Exception as e:
                print(f"❌ Erro ao processar lote {i+1}: {str(e)}")
                continue

        self.memory = accumulated_profile
        self.memory.setdefault("reflexoes", [])
        self.memory.setdefault("planos", [])
        self.memory.setdefault("elogios", [])
        print(f"✅ Memória inicial gerada para Rui: {json.dumps(self.memory, indent=2, ensure_ascii=False)}")

    def analyze(self, conversation_turns: list) -> dict:
        batches = self._split_into_batches(conversation_turns)
        print(f"📝 Analisando conversa para Rui em {len(batches)} lotes.")

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
            print(f"📋 Prompt para lote {i+1} (análise): {prompt[:500]}... (truncado para log)")
            print(f"📏 Tamanho estimado do prompt: ~{len(prompt.split()) // 0.75} tokens")
            print(f"📜 Conversa enviada (lote {i+1}): {conversation_text[:500]}... (truncado para log)")

            try:
                feedback = self.ask_model(prompt)
                feedback_text = feedback.get("choices", [{}])[0].get("text", "").strip()
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

                # Atualizar valores e gatilhos
                feedback_lower = feedback_text.lower()
                if "stress" in feedback_lower and "stress do trabalho" not in accumulated_feedback["gatilhos_emocionais"]:
                    accumulated_feedback["gatilhos_emocionais"].append("stress do trabalho")
                if "conexão" in feedback_lower and "conexão" not in accumulated_feedback["valores"]:
                    accumulated_feedback["valores"].append("conexão")

                previous_feedback = feedback_text
                print(f"📤 Feedback parcial Rui (lote {i+1}): {feedback_text}")
            except Exception as e:
                print(f"❌ Erro ao processar lote {i+1} (análise): {str(e)}")
                continue

        self.memory["reflexoes"].extend(accumulated_feedback["reflexoes"])
        self.memory["planos"].extend(accumulated_feedback["planos"])
        self.memory["elogios"].extend(accumulated_feedback["elogios"])
        self.memory["valores"] = list(set(self.memory["valores"] + accumulated_feedback["valores"]))
        self.memory["gatilhos_emocionais"] = list(set(self.memory["gatilhos_emocionais"] + accumulated_feedback["gatilhos_emocionais"]))
        
        print(f"✅ Feedback acumulado para Rui: {json.dumps(accumulated_feedback, indent=2, ensure_ascii=False)}")
        return accumulated_feedback

    def format_conversation(self, turns: list) -> str:
        formatted = ""
        for turn in turns:
            for msg in turn:
                sender = msg.get("sender_name", "Desconhecido")
                content = msg.get("content", "")
                timestamp = datetime.fromtimestamp(msg.get("timestamp_ms", 0) / 1000).strftime("%Y-%m-%d %H:%M:%S")
                reactions = msg.get("reactions", [])
                reaction_str = f" [reagiu com {', '.join(r['reaction'] for r in reactions)}]" if reactions else ""
                formatted += f"[{timestamp}] {sender}: {content}{reaction_str}\n"
        return formatted.strip()

    def generate_prompt(self, conversation_text: str, previous_feedback: str, batch_number: int, total_batches: int) -> str:
        personalidade = self.memory.get("personalidade", "Indefinida")
        valores = ", ".join(self.memory.get("valores", []))
        gatilhos = ", ".join(self.memory.get("gatilhos_emocionais", []))
        return f"""
        Tu és o Rui. A tua personalidade é {personalidade}. Valorizas {valores}. Os teus gatilhos emocionais incluem {gatilhos}.
        A partir desta parte da conversa (lote {batch_number}/{total_batches}), gera um relatório emocional detalhado focando-te no Rui.
        Inclui:
        - Reflexões emocionais: como me sinto sobre o que foi dito (ex.: "Sinto-me frustrado quando...").
        - Planos futuros: o que pretendo fazer ou mudar (ex.: "Gostava de falar mais sobre...").
        - Gestos de carinho: momentos em que mostrei ou recebi afeto (ex.: "A Maria disse algo que me fez sorrir").
        Fala na primeira pessoa e usa exemplos específicos da conversa.
        Considera o feedback anterior para manter consistência:

        Feedback anterior:
        {previous_feedback}

        Conversa:
        {conversation_text}
        Relatório emocional:
        """

    def _split_into_batches(self, turns: list) -> list:
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
        print(f"📡 Enviando requisição para API: {data['model']}, max_tokens={data['max_tokens']}")
        try:
            response = requests.post(f"{self.model_url}/v1/completions", headers=headers, json=data)
            response.raise_for_status()
            print(f"✅ Resposta recebida: {response.status_code}")
            return response.json()
        except requests.HTTPError as e:
            print(f"❌ Erro HTTP: {response.status_code} - {response.text}")
            raise Exception(f"Erro ao chamar a API do modelo: {response.status_code} - {response.text}")
        except requests.RequestException as e:
            print(f"❌ Erro de rede: {str(e)}")
            raise Exception(f"Erro ao chamar a API do modelo: {str(e)}")