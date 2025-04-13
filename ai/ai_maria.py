from datetime import datetime
import json
import re
import math
from ai.ai_base import BaseAI

class MariaAI(BaseAI):
    def __init__(self, memory: dict, model_url: str):
        super().__init__(memory, model_url)
        self.MAX_TOKENS_PER_BATCH = 2000

    def format_conversation(self, blocks: list) -> str:
        formatted = ""
        for block in blocks:
            input_msg = block["input"]
            response_msg = block["response"]
            input_timestamp = datetime.fromtimestamp(input_msg["timestamp_ms"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
            response_timestamp = datetime.fromtimestamp(response_msg["timestamp_ms"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
            input_sender = "Eu" if input_msg["sender"] == "Maria" else "Rui"
            response_sender = "Eu" if response_msg["sender"] == "Maria" else "Rui"
            input_message = self.fix_encoding(input_msg['message'])
            response_message = self.fix_encoding(response_msg['message'])
            formatted += f"[{input_timestamp}] {input_sender}: {input_message}\n"
            formatted += f"[{response_timestamp}] {response_sender}: {response_message}\n"
        return formatted.strip()

    def generate_initial_memory(self, blocks: list):
        batches = self._split_into_batches(blocks)
        for i, batch in enumerate(batches):
            conversation_text = self.format_conversation(batch)
            prompt = f"""
Tu és um psicólogo analisando Maria com base nesta conversa (lote {i+1}/{len(batches)}).
- 'Eu' refere-se a Maria; 'Rui' é a outra pessoa.
- Foca nas mensagens de Maria para entender sua personalidade, valores, emoções e dinâmicas relacionais.
- Usa mensagens de Rui como contexto.
- Interpreta o tom, subtexto e padrões para inferir estados psicológicos.
- Usa a data '2025-04-12'.
- Não inclua citações diretas como "Maria disse X" ou "Rui disse Y".
- Retorna SOMENTE um JSON válido com:
  - personality: {{"traits": ["string"], "description": "string"}}
  - core_values: [{{"value": "string", "manifestation": "string", "evidence": "string"}}]
  - emotional_patterns: [{{"emotion": "string", "frequency": "string", "triggers": ["string"], "impact": "string"}}]
  - relational_dynamics: {{"strengths": ["string"], "challenges": ["string"], "patterns": ["string"]}}
  - struggles: [{{"issue": "string", "context": "string", "reflection": "string"}}]
  - memories: [{{"date": "YYYY-MM-DD", "event": "string", "emotion": "string", "reflection": "string"}}]
  - reflexoes: [{{"data": "YYYY-MM-DD", "texto": "string"}}]
  - planos: [{{"data": "YYYY-MM-DD", "texto": "string"}}]
  - elogios: [{{"data": "YYYY-MM-DD", "texto": "string"}}]
  - relationship_metrics: [{{"metric": "string", "value": "string", "context": "string"}}]
  - insights: [{{"aspecto": "string", "texto": "string"}}]
- Exemplo de JSON válido:
  {{
    "personality": {{"traits": ["empática"], "description": "Sou empática mas sensível."}},
    "core_values": [{{"value": "compreensão", "manifestation": "Quero ouvir os outros.", "evidence": "Mostrei interesse genuíno."}}],
    "emotional_patterns": [{{"emotion": "preocupação", "frequency": "ocasional", "triggers": ["Tristeza de Rui"], "impact": "Quero ajudar."}}],
    "relational_dynamics": {{"strengths": ["Empatia"], "challenges": ["Ansiedade"], "patterns": ["Busco apoiar"]}},
    "struggles": [{{"issue": "Ansiedade", "context": "Emoções de Rui", "reflection": "Preciso me equilibrar."}}],
    "memories": [{{"date": "2025-04-12", "event": "Conversa com Rui", "emotion": "Empatia", "reflection": "Senti vontade de ajudar."}}],
    "reflexoes": [{{"data": "2025-04-12", "texto": "Eu fico mais calma quando ouço Rui com atenção."}}],
    "planos": [{{"data": "2025-04-12", "texto": "Quero perguntar mais a Rui sobre seus sentimentos."}}],
    "elogios": [{{"data": "2025-04-12", "texto": "Fui atenciosa na conversa de hoje."}}],
    "relationship_metrics": [{{"metric": "Empatia", "value": "Alta", "context": "Mostrei interesse por Rui"}}],
    "insights": [{{"aspecto": "Empatia", "texto": "Surge quando Rui expressa emoções."}}]
  }}
- Usa aspas duplas para TODAS chaves e valores de string.
- Garante sintaxe JSON válida, sem texto fora do JSON.
- Responda em UTF-8, preservando caracteres portugueses (ex.: 'não', 'sensível').
Conversa:
{conversation_text}
"""
            try:
                response = self._call_model_api(prompt)
                profile_text = response.get("choices", [{}])[0].get("text", "{}").strip()
                profile_text = self._clean_json(profile_text)
                print(f"📜 Perfil gerado para Maria, lote {i+1}: {profile_text[:200]}...")
                try:
                    profile_data = json.loads(profile_text)
                    if profile_data:
                        self.update_memory(profile_data)
                    else:
                        print(f"⚠️ JSON vazio para Maria, lote {i+1}, pulando atualização.")
                except json.JSONDecodeError as e:
                    print(f"❌ Erro de JSON no lote {i+1}: {str(e)}")
                    continue
            except Exception as e:
                print(f"❌ Erro no lote {i+1}: {str(e)}")
                continue

        print(f"🔍 Estado de core_values antes da deduplicação: {self.memory['core_values']}")
        # Deduplicate safely
        valid_core_values = []
        for item in self.memory["core_values"]:
            if isinstance(item, dict) and "value" in item:
                valid_core_values.append(item)
            else:
                print(f"🗑️ Entrada inválida em deduplicação de core_values: {item}")
        self.memory["core_values"] = list({v["value"]: v for v in valid_core_values}.values())
        self.memory["personality"]["traits"] = list(set(self.memory["personality"]["traits"]))
        self.memory["emotional_patterns"] = list({e["emotion"]: e for e in self.memory["emotional_patterns"] if isinstance(e, dict) and "emotion" in e}.values())

    def analyze(self, blocks: list) -> dict:
        batches = self._split_into_batches(blocks)
        accumulated_feedback = {k: v.copy() for k, v in self.MEMORY_SCHEMA.items()}
        for i, batch in enumerate(batches):
            conversation_text = self.format_conversation(batch)
            print(f"📋 Texto analisado para Maria, lote {i+1}:\n{conversation_text}\n")
            prompt = f"""
Tu és a Maria, refletindo sobre esta conversa (lote {i+1}/{len(batches)}).
- 'Eu' refere-se a Maria; 'Rui' é a outra pessoa.
- Fala na primeira pessoa para reflexões, planos e elogios.
- Usa mensagens de Rui como contexto.
- Usa a data '2025-04-12'.
- Não inclua citações diretas como "Maria disse X" ou "Rui disse Y".
- Reflexões devem conter 'eu', 'meu' ou 'minha' e expressar pensamentos ou sentimentos pessoais.
- Elogios devem ser meus aspectos positivos, como "Fui atenciosa ao ouvir Rui".
- Baseia-te no perfil psicológico: {json.dumps(self.memory, ensure_ascii=False)}.
- Retorna SOMENTE um JSON válido com:
  - personality: {{"traits": ["string"], "description": "string"}}
  - core_values: [{{"value": "string", "manifestation": "string", "evidence": "string"}}]
  - emotional_patterns: [{{"emotion": "string", "frequency": "string", "triggers": ["string"], "impact": "string"}}]
  - relational_dynamics: {{"strengths": ["string"], "challenges": ["string"], "patterns": ["string"]}}
  - struggles: [{{"issue": "string", "context": "string", "reflection": "string"}}]
  - memories: [{{"date": "YYYY-MM-DD", "event": "string", "emotion": "string", "reflection": "string"}}]
  - reflexoes: [{{"data": "YYYY-MM-DD", "texto": "string"}}]
  - planos: [{{"data": "YYYY-MM-DD", "texto": "string"}}]
  - elogios: [{{"data": "YYYY-MM-DD", "texto": "string"}}]
  - relationship_metrics: [{{"metric": "string", "value": "string", "context": "string"}}]
  - insights: [{{"aspecto": "string", "texto": "string"}}]
- Exemplo de JSON válido:
  {{
    "personality": {{"traits": ["empática"], "description": "Sou empática mas sensível."}},
    "core_values": [{{"value": "compreensão", "manifestation": "Quero ouvir os outros.", "evidence": "Mostrei interesse genuíno."}}],
    "emotional_patterns": [{{"emotion": "preocupação", "frequency": "ocasional", "triggers": ["Tristeza de Rui"], "impact": "Quero ajudar."}}],
    "relational_dynamics": {{"strengths": ["Empatia"], "challenges": ["Ansiedade"], "patterns": ["Busco apoiar"]}},
    "struggles": [{{"issue": "Ansiedade", "context": "Emoções de Rui", "reflection": "Preciso me equilibrar."}}],
    "memories": [{{"date": "2025-04-12", "event": "Conversa com Rui", "emotion": "Empatia", "reflection": "Senti vontade de ajudar."}}],
    "reflexoes": [{{"data": "2025-04-12", "texto": "Eu fico mais calma quando ouço Rui com atenção."}}],
    "planos": [{{"data": "2025-04-12", "texto": "Quero perguntar mais a Rui sobre seus sentimentos."}}],
    "elogios": [{{"data": "2025-04-12", "texto": "Fui atenciosa na conversa de hoje."}}],
    "relationship_metrics": [{{"metric": "Empatia", "value": "Alta", "context": "Mostrei interesse por Rui"}}],
    "insights": [{{"aspecto": "Empatia", "texto": "Surge quando Rui expressa emoções."}}]
  }}
- Usa aspas duplas para TODAS chaves e valores de string.
- Garante sintaxe JSON válida, sem texto fora do JSON.
- Responda em UTF-8, preservando caracteres portugueses (ex.: 'não', 'sensível').
Conversa:
{conversation_text}
"""
            try:
                feedback = self._call_model_api(prompt)
                feedback_text = feedback.get("choices", [{}])[0].get("text", "{}").strip()
                feedback_text = self._clean_json(feedback_text)
                print(f"📜 Resposta parseada para Maria, lote {i+1}: {feedback_text[:200]}...")
                try:
                    data = json.loads(feedback_text)
                    filtered_reflexoes = []
                    for reflexao in data.get("reflexoes", []):
                        texto = reflexao.get("texto", "").lower()
                        if ("rui disse" not in texto and
                            any(word in texto for word in ["eu ", "meu ", "minha "]) and
                            reflexao.get("data", "") == "2025-04-12"):
                            filtered_reflexoes.append(reflexao)
                        else:
                            print(f"🗑️ Reflexão descartada: {reflexao}")
                    data["reflexoes"] = filtered_reflexoes
                    filtered_elogios = []
                    for elogio in data.get("elogios", []):
                        texto = elogio.get("texto", "").lower()
                        if ("rui disse" not in texto and
                            any(word in texto for word in ["eu ", "meu ", "minha ", "fui "]) and
                            elogio.get("data", "") == "2025-04-12"):
                            filtered_elogios.append(elogio)
                        else:
                            print(f"🗑️ Elogio descartado: {elogio}")
                    data["elogios"] = filtered_elogios
                    for key in accumulated_feedback:
                        if key in data:
                            if isinstance(accumulated_feedback[key], list):
                                accumulated_feedback[key].extend(data[key])
                            elif isinstance(accumulated_feedback[key], dict):
                                accumulated_feedback[key].update(data[key])
                except json.JSONDecodeError as e:
                    print(f"❌ Erro de JSON no lote {i+1}: {str(e)}")
                    continue
            except Exception as e:
                print(f"❌ Erro no lote {i+1}: {str(e)}")
                continue

        self.update_memory(accumulated_feedback)
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