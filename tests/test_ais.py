import unittest
from ai.ai_rui import RuiAI
from ai.ai_maria import MariaAI
from ai.ai_relational import RelationalAI

class TestAIModules(unittest.TestCase):
    def setUp(self):
        # Memórias iniciais vazias
        self.rui_memory = {"insights": [], "frustrations": []}
        self.maria_memory = {"insights": [], "frustrations": []}
        self.relational_memory = {"weekly_reports": []}
        # Simulação de blocos de interação (exemplo simplificado)
        self.interaction_blocks = [
            {"input": {"sender": "Maria", "message": "Estou cansada hoje."},
             "response": {"sender": "Rui", "message": "Espero que o teu dia melhore."}},
            {"input": {"sender": "Rui", "message": "Tenho estado muito estressado."},
             "response": {"sender": "Maria", "message": "É importante descansares."}}
        ]

    def test_rui_ai_feedback(self):
        rui_ai = RuiAI(memory=self.rui_memory)
        feedback = rui_ai.analyze_conversation(self.interaction_blocks)
        self.assertIsInstance(feedback, list)
        self.assertTrue(len(feedback) > 0)

    def test_maria_ai_feedback(self):
        maria_ai = MariaAI(memory=self.maria_memory)
        feedback = maria_ai.analyze_conversation(self.interaction_blocks)
        self.assertIsInstance(feedback, list)
        self.assertTrue(len(feedback) > 0)

    def test_relational_ai_report(self):
        rui_ai = RuiAI(memory=self.rui_memory)
        maria_ai = MariaAI(memory=self.maria_memory)
        relational_ai = RelationalAI(memory=self.relational_memory)
        rui_feedback = rui_ai.analyze_conversation(self.interaction_blocks)
        maria_feedback = maria_ai.analyze_conversation(self.interaction_blocks)
        report = relational_ai.generate_report(rui_feedback, maria_feedback)
        self.assertIsInstance(report, str)
        self.assertIn("RELATÓRIO SEMANAL", report)

if __name__ == '__main__':
    unittest.main()
