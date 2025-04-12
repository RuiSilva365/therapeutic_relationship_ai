from datetime import datetime
import json
import re
import math
from ai.ai_base import BaseAI

class RuiAI(BaseAI):
    def __init__(self, memory: dict, model_url: str):
        super().__init__(memory, model_url)
        self.memory = memory or {
            "personalidade": "Indefinida", 
            "valores": [], 
            "gatilhos_emocionais": [], 
            "reflexoes": [], 
            "planos": [], 
            "elogios": [],
            "relationship_metrics": []
        }
        self.MAX_TOKENS_PER_BATCH = 2000

    def format_conversation(self, blocks: list) -> str:
        formatted = ""
        for block in blocks:
            input_msg = block["input"]
            response_msg = block["response"]
            input_timestamp = datetime.fromtimestamp(input_msg["timestamp_ms"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
            response_timestamp = datetime.fromtimestamp(response_msg["timestamp_ms"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
            input_sender = "Eu" if input_msg["sender"] == "Rui" else "Maria"
            response_sender = "Eu" if response_msg["sender"] == "Rui" else "Maria"
            formatted += f"[{input_timestamp}] {input_sender}: {input_msg['message']}\n"
            formatted += f"[{response_timestamp}] {response_sender}: {response_msg['message']}\n"
        return formatted.strip()

    def generate_initial_memory(self, blocks: list):
        batches = self._split_into_batches(blocks)
        print(f"📝 Gerando memória inicial para Rui em {len(batches)} lotes.")

        accumulated_profile = {
            "personalidade": "Indefinida",
            "valores": [],
            "gatilhos_emocionais": [],
            "reflexoes": [],
            "planos": [],
            "elogios": []
        }

        for i, batch in enumerate(batches):
            conversation_text = self.format_conversation(batch)
            print(f"📋 Texto analisado para Rui, lote {i+1}:\n{conversation_text}\n")
            prompt = f"""
Analisa esta conversa entre Rui e Maria (lote {i+1}/{len(batches)}).
- 'Eu' refere-se a Rui; 'Maria' é a outra pessoa.
- Foca apenas nas mensagens de Rui para personalidade, valores e gatilhos.
- Usa a data '2025-04-12' para reflexões.
Retorna SOMENTE um JSON com:
- personalidade: str (ex.: "introspectivo", "Indefinida" se não claro)
- valores: lista de str (ex.: ["conexão", "honestidade"])
- gatilhos_emocionais: lista de str (ex.: ["stress", "rejeição"])
Sem texto fora do JSON. Use exemplos da conversa. Responda em UTF-8.
Conversa:
{conversation_text}
"""
            try:
                response = self._call_model_api(prompt)
                profile_text = response.get("choices", [{}])[0].get("text", "{}").strip()
                profile_text = self._clean_json(profile_text)
                print(f"📜 Resposta parseada para Rui, lote {i+1}: {profile_text[:200]}...")
                try:
                    profile_data = json.loads(profile_text)
                except json.JSONDecodeError:
                    json_match = re.search(r'\{[\s\S]*\}', profile_text)
                    if json_match:
                        profile_data = json.loads(json_match.group(0))
                    else:
                        print(f"❌ Sem JSON válido no lote {i+1}")
                        continue
                if profile_data.get("personalidade") and profile_data["personalidade"] != "Indefinida":
                    accumulated_profile["personalidade"] = profile_data["personalidade"]
                accumulated_profile["valores"] = list(set(accumulated_profile["valores"] + [
                    str(v) for v in profile_data.get("valores", []) if isinstance(v, str)
                ]))
                accumulated_profile["gatilhos_emocionais"] = list(set(accumulated_profile["gatilhos_emocionais"] + [
                    str(v) for v in profile_data.get("gatilhos_emocionais", []) if isinstance(v, str)
                ]))
            except Exception as e:
                print(f"❌ Erro no lote {i+1}: {str(e)}")
                continue

        self.memory.update(accumulated_profile)
        self.memory.setdefault("reflexoes", [])
        self.memory.setdefault("planos", [])
        self.memory.setdefault("elogios", [])
        self.memory.setdefault("relationship_metrics", [])

    def analyze(self, blocks: list) -> dict:
        batches = self._split_into_batches(blocks)
        accumulated_feedback = {
            "reflexoes": [],
            "planos": [],
            "elogios": [],
            "valores": self.memory.get("valores", []),
            "gatilhos_emocionais": self.memory.get("gatilhos_emocionais", [])
        }

        for i, batch in enumerate(batches):
            conversation_text = self.format_conversation(batch)
            print(f"📋 Texto analisado para Rui, lote {i+1}:\n{conversation_text}\n")
            prompt = f"""
Tu és o Rui. Analisa esta conversa (lote {i+1}/{len(batches)}).
- 'Eu' refere-se a Rui; 'Maria' é a outra pessoa.
- Fala na primeira pessoa apenas para mensagens de Rui.
- Usa mensagens de Maria como contexto.
- Usa a data '2025-04-12' para reflexões, planos e elogios.
Retorna SOMENTE um JSON com:
- reflexoes: lista de {{data: "YYYY-MM-DD", texto: str}}
- planos: lista de {{data: "YYYY-MM-DD", texto: str}}
- elogios: lista de {{data: "YYYY-MM-DD", texto: str}}
- valores: lista de str
- gatilhos_emocionais: lista de str
Sem texto fora do JSON. Use exemplos da conversa. Responda em UTF-8.
Conversa:
{conversation_text}
"""
            try:
                feedback = self._call_model_api(prompt)
                feedback_text = feedback.get("choices", [{}])[0].get("text", "{}").strip()
                feedback_text = self._clean_json(feedback_text)
                print(f"📜 Resposta parseada para Rui, lote {i+1}: {feedback_text[:200]}...")
                try:
                    data = json.loads(feedback_text)
                except json.JSONDecodeError:
                    json_match = re.search(r'\{[\s\S]*\}', feedback_text)
                    if json_match:
                        data = json.loads(json_match.group(0))
                    else:
                        print(f"❌ Sem JSON válido no lote {i+1}")
                        continue
                accumulated_feedback["reflexoes"].extend([
                    r for r in data.get("reflexoes", []) if isinstance(r, dict) and "data" in r and "texto" in r
                ])
                accumulated_feedback["planos"].extend([
                    p for p in data.get("planos", []) if isinstance(p, dict) and "data" in p and "texto" in p
                ])
                accumulated_feedback["elogios"].extend([
                    e for e in data.get("elogios", []) if isinstance(e, dict) and "data" in e and "texto" in e
                ])
                accumulated_feedback["valores"] = list(set(accumulated_feedback["valores"] + [
                    str(v) for v in data.get("valores", []) if isinstance(v, str)
                ]))
                accumulated_feedback["gatilhos_emocionais"] = list(set(accumulated_feedback["gatilhos_emocionais"] + [
                    str(v) for v in data.get("gatilhos_emocionais", []) if isinstance(v, str)
                ]))
            except Exception as e:
                print(f"❌ Erro no lote {i+1}: {str(e)}")
                continue

        self.memory["reflexoes"].extend(accumulated_feedback["reflexoes"])
        self.memory["planos"].extend(accumulated_feedback["planos"])
        self.memory["elogios"].extend(accumulated_feedback["elogios"])
        self.memory["valores"] = list(set(self.memory["valores"] + accumulated_feedback["valores"]))
        self.memory["gatilhos_emocionais"] = list(set(self.memory["gatilhos_emocionais"] + accumulated_feedback["gatilhos_emocionais"]))
        return accumulated_feedback

    def _split_into_batches(self, blocks: list) -> list:
        batches = []
        current_batch = []
        current_tokens = 0
        words_per_token = 0.75

        for block in blocks:
            block_text = self.format_conversation([block])
            token_count = math.ceil(len(block_text.split()) / words_per_token)
            if current_tokens + token_count > self.MAX_TOKENS_PER_BATCH:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [block]
                current_tokens = token_count
            else:
                current_batch.append(block)
                current_tokens += token_count
        if current_batch:
            batches.append(current_batch)
        return batches