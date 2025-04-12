from datetime import datetime
import json
import re
from ai.ai_base import BaseAI

class RelationalAI(BaseAI):
    def __init__(self, memory: dict, model_url: str):
        super().__init__(memory, model_url)
        self.memory = memory or {"dinamicas_relacionais": []}

    def generate_feedback(self, rui_feedback: dict, maria_feedback: dict) -> dict:
        prompt = self._construct_prompt(rui_feedback, maria_feedback)
        response = self._call_model_api(prompt=prompt, max_tokens=3000, temperature=0.3)
        feedback_text = response.get("choices", [{}])[0].get("text", "{}").strip()
        print(f"📜 Resposta relacional: {feedback_text[:200]}...")

        try:
            json_match = re.search(r'\{[\s\S]*\}', feedback_text)
            if json_match:
                feedback_data = json.loads(json_match.group(0))
                positivos = feedback_data.get("positivos", [])
                negativos = feedback_data.get("negativos", [])
                conclusao = feedback_data.get("conclusao", "")
            else:
                raise ValueError("Nenhum JSON encontrado")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Erro ao parsear feedback relacional: {str(e)}")
            positivos = []
            negativos = []
            conclusao = feedback_text
            for line in feedback_text.split("\n"):
                line = line.strip()
                if "positivo" in line.lower():
                    positivos.append(line)
                elif "negativo" in line.lower():
                    negativos.append(line)

        report = {
            "data": datetime.now().strftime("%Y-%m-%d"),
            "positivos": positivos,
            "negativos": negativos,
            "conclusao": conclusao
        }
        self.memory["dinamicas_relacionais"].append(report)

        return {
            "feedback": feedback_text,
            "analysis": report
        }

    def _construct_prompt(self, rui_feedback: dict, maria_feedback: dict) -> str:
        prompt = "Análise Relacional:\n"
        prompt += "Feedback do Rui (reflete apenas suas próprias mensagens):\n"
        for key, values in rui_feedback.items():
            prompt += f"{key}: {json.dumps(values, ensure_ascii=False)}\n"
        prompt += "\nFeedback da Maria (reflete apenas suas próprias mensagens):\n"
        for key, values in maria_feedback.items():
            prompt += f"{key}: {json.dumps(values, ensure_ascii=False)}\n"
        prompt += """
Tu és um analista emocional especializado em relações amorosas. Com base nos feedbacks fornecidos:
- Rui’s reflections, plans, and elogios são baseados apenas nas suas mensagens.
- Maria’s reflections, plans, and elogios são baseados apenas nas suas mensagens.
Gera um relatório relacional em JSON com:
- positivos: lista de str (aspectos positivos da relação)
- negativos: lista de str (aspectos negativos da relação)
- conclusao: str (resumo da dinâmica relacional)
Inclui tua percepção emocional sobre a relação (como te faz sentir, o que pensas).
Retorna SOMENTE o JSON, sem texto adicional.
"""
        return prompt