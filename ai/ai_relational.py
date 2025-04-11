from datetime import datetime  # Adicionar esta importaçãoimport json
import os
import requests  # type: ignore
from ai.ai_base import BaseAI

class RelationalAI(BaseAI):
    def __init__(self, memory: dict, model_url: str):
        """
        Inicializa a IA Relational com memória e a URL completa da API.
        :param memory: Dicionário que armazena a memória da IA (ex: {"weekly_reports": []}).
        :param model_url: URL completa da API (ex: "http://192.168.56.1:1234/v1/completions").
        """
        super().__init__(memory, model_url)
        self.memory = memory or {"weekly_reports": []}

    def generate_feedback(self, rui_feedback: dict, maria_feedback: dict) -> dict:
        
        
        prompt = self._construct_prompt(rui_feedback, maria_feedback)
        response = self._call_model_api(prompt)
        feedback_text = response.get("choices", [{}])[0].get("text", "").strip()
        
        # Extrair positivos, negativos e conclusão (simplificado)
        positivos = []
        negativos = []
        conclusao = ""
        for line in feedback_text.split("\n"):
            line = line.strip()
            if "pontos positivos" in line.lower():
                positivos.append(line)
            elif "pontos negativos" in line.lower():
                negativos.append(line)
            elif "resumo" in line.lower() or "conclusão" in line.lower():
                conclusao = line
        
        # Atualizar memória
        self.memory.setdefault("dinamicas_relacionais", []).append({
            "data": datetime.now().strftime("%Y-%m-%d"),
            "positivos": positivos,
            "negativos": negativos,
            "conclusao": conclusao
        })
        
        return {
            "feedback": feedback_text,
            "analysis": "Análise não separada."
        }

    def _construct_prompt(self, rui_feedback: dict, maria_feedback: dict) -> str:
        """
        Constrói o prompt integrando os feedbacks de Rui e Maria para obter uma análise relacional.
        :param rui_feedback: Feedback da RuiAI.
        :param maria_feedback: Feedback da MariaAI.
        :return: Prompt formatado.
        """
        prompt = "Análise Relacional:\n"
        prompt += "Feedback do Rui:\n"
        for key, values in rui_feedback.items():
            if isinstance(values, (list, tuple)):
                prompt += f"{key}: {', '.join(str(item) for item in values)}\n"
            else:
                prompt += f"{key}: {values}\n"
        prompt += "\nFeedback da Maria:\n"
        for key, values in maria_feedback.items():
            if isinstance(values, (list, tuple)):
                prompt += f"{key}: {', '.join(str(item) for item in values)}\n"
            else:
                prompt += f"{key}: {values}\n"
        prompt += "\n Tu és um analista emocional muito sensível e especializado em relações amorosas. Com base nos perfis psicológicos de cada um dos intervenientes, analisa o feedback emocional de ambos e gera um relatório emocional relacional. O relatório deve incluir os pontos positivos e negativos de cada um(caso haja) na ultima semana, bem como a dinâmica relacional entre eles. Não te esqueças de incluir a tua percepção sobre a conversa (como te fez sentir e o que pensas sobre isso). Tens de ser honesto e direto.\n"
        return prompt
