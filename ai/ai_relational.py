from datetime import datetime
import json
import re
from ai.ai_base import BaseAI

class RelationalAI(BaseAI):
    def __init__(self, memory: dict, model_url: str):
        super().__init__(memory, model_url)
        # Ensure required keys with correct types
        self.memory.setdefault("rui_profile", {})
        self.memory.setdefault("maria_profile", {})
        # Force relational_dynamics to be a list
        if not isinstance(self.memory.get("relational_dynamics"), list):
            self.memory["relational_dynamics"] = []
        print(f"🧠 Memória inicializada para RelationalAI: {json.dumps(self.memory, ensure_ascii=False)[:200]}...")

    def generate_feedback(self, rui_feedback: dict, maria_feedback: dict) -> dict:
        prompt = self._construct_prompt(rui_feedback, maria_feedback)
        token_estimate = len(prompt.split()) // 0.75
        print(f"📏 Gerando relatório relacional com ~{token_estimate} tokens")
        response = self._call_model_api(prompt=prompt, max_tokens=1000, temperature=0.3)
        feedback_text = response.get("choices", [{}])[0].get("text", "{}").strip()
        print(f"📜 Resposta relacional: {feedback_text[:200]}...")

        try:
            json_match = re.search(r'\{[\s\S]*\}', feedback_text)
            if json_match:
                feedback_data = json.loads(json_match.group(0))
                strengths = feedback_data.get("strengths", feedback_data.get("positivos", []))
                challenges = feedback_data.get("challenges", feedback_data.get("negativos", []))
                advice = feedback_data.get("advice", feedback_data.get("conselhos", []))
            else:
                raise ValueError("Nenhum JSON encontrado")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Erro ao parsear feedback relacional: {str(e)}")
            strengths = []
            challenges = []
            advice = []

        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "strengths": strengths[:3],
            "challenges": challenges[:3],
            "advice": advice[:3]
        }
        print(f"🧠 Memória antes de atualizar: {json.dumps(self.memory, ensure_ascii=False)[:200]}...")
        self.memory["relational_dynamics"].append(report)
        self.memory["rui_profile"] = rui_feedback | self.memory["rui_profile"]
        self.memory["maria_profile"] = maria_feedback | self.memory["maria_profile"]
        print(f"🧠 Memória após atualizar: {json.dumps(self.memory, ensure_ascii=False)[:200]}...")

        return report

    def _construct_prompt(self, rui_feedback: dict, maria_feedback: dict) -> str:
        prompt = f"""
You are an emotional analyst specializing in romantic relationships. Based on the provided feedback:
- Rui's reflections: {json.dumps(rui_feedback.get("recent_reflections", []), ensure_ascii=False)}
- Maria's reflections: {json.dumps(maria_feedback.get("recent_reflections", []), ensure_ascii=False)}
- Rui's profile: {json.dumps(self.memory.get("rui_profile", {}), ensure_ascii=False)}
- Maria's profile: {json.dumps(self.memory.get("maria_profile", {}), ensure_ascii=False)}
Generate a relational report in JSON with:
- strengths: list of strings (positive aspects of the relationship)
- challenges: list of strings (negative aspects of the relationship)
- advice: list of strings (suggestions to improve the relationship)
- Maximum of 3 items per list.
Return ONLY the JSON with keys "strengths", "challenges", and "advice", in English, with no additional text.
Example:
{{
  "strengths": ["Mutual trust", "Open communication"],
  "challenges": ["Emotional distance", "Lack of time together"],
  "advice": ["Discuss feelings openly", "Plan quality time"]
}}
"""
        return prompt