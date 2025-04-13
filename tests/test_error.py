import json
from ai.ai_rui import RuiAI
from ai.ai_maria import MariaAI

# Use the problematic block
blocks = [
    {
        "input": {"sender": "Rui", "timestamp_ms": 1729281008866, "message": "Nao Ã©?ð"},
        "response": {"sender": "Maria", "timestamp_ms": 1729284341951, "message": "???"}
    }
]

MODEL_URL = "http://192.168.56.1:1234"
rui_ai = RuiAI(memory=None, model_url=MODEL_URL)
maria_ai = MariaAI(memory=None, model_url=MODEL_URL)

print("📝 Gerando memória para Rui...")
rui_ai.generate_initial_memory(blocks)
with open('data/rui_memory_test.json', 'w', encoding='utf-8') as f:
    json.dump(rui_ai.memory, f, indent=2, ensure_ascii=False)

print("📝 Gerando memória para Maria...")
maria_ai.generate_initial_memory(blocks)
with open('data/maria_memory_test.json', 'w', encoding='utf-8') as f:
    json.dump(maria_ai.memory, f, indent=2, ensure_ascii=False)