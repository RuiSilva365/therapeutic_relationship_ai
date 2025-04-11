import json
import os
from datetime import datetime, timedelta
from ai.ai_rui import RuiAI
from ai.ai_maria import MariaAI
from ai.ai_relational import RelationalAI

# === UTILS ===
def load_memory(path):
    """Carrega a memória de um arquivo (JSON ou texto simples) ou retorna None se não existir."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            try:
                return json.loads(content)  # Tenta parsear como JSON
            except json.JSONDecodeError:
                return {"raw_text": content}  # Se não for JSON, armazena como texto bruto
    return None

def save_memory(path, data):
    """Salva a memória no arquivo em formato JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_conversations(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_report(report):
    today = datetime.today().strftime('%Y-%m-%d')
    path = f'reports/report_{today}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

def filter_last_week(messages):
    one_week_ago = datetime.now() - timedelta(days=7)
    filtered_messages = []
    for msg in messages:
        if isinstance(msg, dict) and 'timestamp_ms' in msg:
            timestamp_ms = msg['timestamp_ms']
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
            if timestamp > one_week_ago:
                filtered_messages.append(msg)
        else:
            print(f"Mensagem ignorada (não contém 'timestamp_ms'): {msg}")
    return filtered_messages

def group_conversation_turns(messages):
    grouped = []
    buffer = []
    for msg in messages:
        buffer.append(msg)
        if msg['sender_name'] == 'Maria':
            grouped.append(buffer)
            buffer = []
    if buffer:
        grouped.append(buffer)
    return grouped

# === INÍCIO DO SCRIPT ===
MODEL_URL = "http://192.168.56.1:1234"

# Caminhos dos ficheiros
rui_memory_path = 'data/rui_memory'
maria_memory_path = 'data/maria_memory'
relational_memory_path = 'data/relational_memory.json'  # Mantém em JSON por simplicidade

# Carregar memórias existentes
rui_memory = load_memory(rui_memory_path)
maria_memory = load_memory(maria_memory_path)
relational_memory = load_memory(relational_memory_path) or {"dinamicas_relacionais": []}

# Inicializar IAs
ai_rui = RuiAI(memory=rui_memory, model_url=MODEL_URL)
ai_maria = MariaAI(memory=maria_memory, model_url=MODEL_URL)
ai_relational = RelationalAI(memory=relational_memory, model_url=MODEL_URL)

# Carregar mensagens
conversation_data = load_conversations('data/conversation_simulated.json')
messages = conversation_data.get("messages", [])
recent_messages = messages #filter_last_week(messages)

print(f"🔍 Mensagens filtradas: {len(recent_messages)} mensagens na última semana.")
turns = group_conversation_turns(recent_messages)
print(turns)

# Gerar memórias iniciais se não existirem
if rui_memory is None:
    print("📝 Gerando memória inicial para Rui...")
    ai_rui.generate_initial_memory(turns)
    save_memory(rui_memory_path, ai_rui.memory)

if maria_memory is None:
    print("📝 Gerando memória inicial para Maria...")
    ai_maria.generate_initial_memory(turns)
    save_memory(maria_memory_path, ai_maria.memory)

# Análise das IAs
rui_feedback = ai_rui.analyze(turns)
print("\n🤖 [IA RUI]: Este seria o meu feedback da semana:")
print(json.dumps(rui_feedback, indent=2, ensure_ascii=False))

maria_feedback = ai_maria.analyze(turns)
print("\n🤖 [IA MARIA]: Este seria o meu feedback da semana:")
print(json.dumps(maria_feedback, indent=2, ensure_ascii=False))

# Relatório relacional
final_report = ai_relational.generate_feedback(rui_feedback, maria_feedback)
print(final_report)

# Salvar tudo
save_memory(rui_memory_path, ai_rui.memory)
save_memory(maria_memory_path, ai_maria.memory)
save_memory(relational_memory_path, ai_relational.memory)
save_report(final_report)

print("\n📊 RELATÓRIO FINAL GERADO COM SUCESSO:")
print(json.dumps(final_report, indent=2, ensure_ascii=False))
print("\n✅ Memórias atualizadas e relatório salvo.")
print("🔄 Fim do ciclo semanal de feedback.")