from datetime import datetime
import json
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
        max_token_limit = 3000
        conversation_text = ""
        token_estimate = 0
        for block in blocks:
            block_text = self.format_conversation([block])
            block_tokens = len(block_text.split()) // 0.75
            if token_estimate + block_tokens > max_token_limit:
                break
            conversation_text += block_text + "\n"
            token_estimate += block_tokens

        print(f"📏 Gerando perfil para Maria com ~{token_estimate} tokens")
        prompt = f"""
Tu és um psicólogo criando um perfil para Maria com base nesta conversa.
- 'Eu' refere-se a Maria; a outra pessoa é Rui.
- Foca nas mensagens de Maria para entender personalidade, valores e emoções.
- Usa mensagens de Rui como contexto.
- Usa a data '2025-04-12'.
- Retorna SOMENTE um JSON válido com:
  - personality: {{"traits": ["string"], "description": "string"}}
  - core_values: [{{"value": "string", "description": "string"}}]
  - emotional_patterns: [{{"emotion": "string", "triggers": ["string"], "description": "string"}}]
  - relational_dynamics: {{"strengths": ["string"], "challenges": ["string"], "patterns": ["string"]}}
- Máximo de 3 itens por lista para manter concisão.
- Exemplo:
  {{
    "personality": {{"traits": ["empática"], "description": "Sou sensível e preocupada com os outros."}},
    "core_values": [{{"value": "compreensão", "description": "Quero apoiar quem amo."}}],
    "emotional_patterns": [{{"emotion": "ansiedade", "triggers": ["conflito"], "description": "Fico nervosa com discussões."}}],
    "relational_dynamics": {{"strengths": ["afeto"], "challenges": ["insegurança"], "patterns": ["busco proximidade"]}}
  }}
- Usa aspas duplas e UTF-8.
Conversa:
{conversation_text}
"""
        try:
            response = self._call_model_api(prompt, max_tokens=1000)
            profile_text = response.get("choices", [{}])[0].get("text", "{}").strip()
            profile_text = self._clean_json(profile_text)
            profile_data = json.loads(profile_text)
            if profile_data != self.MEMORY_SCHEMA:
                self.update_memory(profile_data)
                print(f"📜 Perfil inicial gerado para Maria: {profile_text[:200]}...")
            else:
                print("⚠️ Perfil vazio para Maria, pulando atualização.")
        except Exception as e:
            print(f"❌ Erro ao gerar perfil inicial para Maria: {str(e)}")

    def analyze(self, blocks: list) -> dict:
        max_token_limit = 3000
        all_reflections = []
        current_blocks = []
        current_tokens = 0

        for block in blocks:
            block_text = self.format_conversation([block])
            block_tokens = len(block_text.split()) // 0.75
            if current_tokens + block_tokens > max_token_limit:
                if current_blocks:
                    reflections = self._process_batch(current_blocks)
                    all_reflections.extend(reflections)
                current_blocks = [block]
                current_tokens = block_tokens
            else:
                current_blocks.append(block)
                current_tokens += block_tokens

        # Process final batch
        if current_blocks:
            reflections = self._process_batch(current_blocks)
            all_reflections.extend(reflections)

        # Deduplicate and limit reflections
        unique_reflections = []
        seen_texts = set()
        for r in all_reflections:
            if r.get("text", "") not in seen_texts and r.get("date", "") == "2025-04-12" and r.get("text", "").strip():
                unique_reflections.append(r)
                seen_texts.add(r["text"])

        data = {"recent_reflections": unique_reflections[:2]}
        if unique_reflections:
            self.update_memory(data)
            print(f"📜 Reflexões para Maria: {json.dumps(data, ensure_ascii=False)[:200]}...")
        else:
            print("⚠️ Reflexões vazias para Maria após validação.")
        return data

    def _process_batch(self, blocks: list) -> list:
        conversation_text = self.format_conversation(blocks)
        token_estimate = len(conversation_text.split()) // 0.75
        print(f"📏 Analisando lote para Maria com ~{token_estimate} tokens")
        print(f"📋 Conversa formatada: {conversation_text[:200]}...")
        prompt = f"""
Tu és Maria, refletindo sobre esta conversa.
- 'Eu' refere-se a Maria; a outra pessoa é Rui.
- Fala na primeira pessoa, expressando sentimentos e pensamentos.
- Baseia-te no meu perfil: {json.dumps(self.memory, ensure_ascii=False)}.
- Usa a data '2025-04-12'.
- Retorna SOMENTE um JSON com:
  - recent_reflections: [{{"date": "YYYY-MM-DD", "text": "string"}}]
- Máximo de 2 reflexões por lote.
- Exemplo:
  {{
    "recent_reflections": [
      {{"date": "2025-04-12", "text": "Senti Rui mais distante hoje, talvez eu precise ser mais clara."}}
    ]
  }}
- Usa aspas duplas e UTF-8.
Conversa:
{conversation_text}
"""
        try:
            response = self._call_model_api(prompt, max_tokens=1000)
            feedback_text = response.get("choices", [{}])[0].get("text", "{}").strip()
            print(f"📜 Resposta bruta para Maria: {feedback_text[:200]}...")
            feedback_text = self._clean_json(feedback_text)
            data = json.loads(feedback_text)
            return data.get("recent_reflections", [])
        except Exception as e:
            print(f"❌ Erro ao analisar lote para Maria: {str(e)}")
            return []