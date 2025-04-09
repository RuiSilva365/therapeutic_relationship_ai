import json
import os
from datetime import datetime, timedelta
from ai.ai_rui import RuiAI
from ai.ai_maria import MariaAI
from ai.ai_relational import RelationalAI


# === UTILS ===
def load_memory(path, default_data=None):
    """Carrega a memória de um arquivo JSON ou retorna dados padrão, se não existir."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    # Se não encontrar o arquivo, retorna os dados padrão fornecidos (se existirem)
    return default_data if default_data is not None else {}

def save_memory(path, data):
    """Salva os dados de memória no arquivo JSON especificado."""
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Garante que a pasta exista
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
            
            # Converter milissegundos para segundos
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
            
            # Filtrar mensagens dentro da última semana
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
        if msg['sender_name'] == 'maria':  # define o fim de um "turno" conversacional
            grouped.append(buffer)
            buffer = []
    if buffer:
        grouped.append(buffer)
    return grouped

# === INÍCIO DO SCRIPT ===

# Definir dados padrão para as memórias, caso os arquivos JSON não existam
default_rui_memory = {}
default_maria_memory = {}
default_relational_memory = {}




# Carregar ou criar memórias
rui_memory = load_memory('ai_memory/rui_memory.json', default_rui_memory)
maria_memory = load_memory('ai_memory/maria_memory.json', default_maria_memory)
relational_memory = load_memory('ai_memory/relational_memory.json', default_relational_memory)


MODEL_URL = "http://192.168.56.1:1234"
ai_rui = RuiAI(memory=rui_memory, model_url=MODEL_URL)
ai_maria = MariaAI(memory=maria_memory, model_url=MODEL_URL)
ai_relational = RelationalAI(memory=relational_memory, model_url=MODEL_URL)

# 2. Carregar mensagens
conversation_data = load_conversations('data/conversation_simulated.json')
# Extraia a lista de mensagens
messages = conversation_data.get("messages", [])
recent_messages = messages

print(f"🔍 Mensagens filtradas: {len(recent_messages)} mensagens na última semana.")
turns = group_conversation_turns(recent_messages)
print(turns)

# 3. IA RUI e IA MARIA analisam suas mensagens
rui_feedback = ai_rui.analyze(turns)
print("\n🤖 [IA RUI]: Este seria o meu feedback da semana:")
print(json.dumps(rui_feedback, indent=2, ensure_ascii=False))

#confirm = input("\nRUI, queres ajustar o feedback? (s/n): ")
#if confirm.lower() == 's':
 #   print("Escreve o teu próprio feedback JSON abaixo:")
  #  custom = input("➡ ")
   # try:
    #    rui_feedback = json.loads(custom)
    #except:
     #   print("❌ JSON inválido, vai ser usado o original.")

maria_feedback = ai_maria.analyze(turns)
print("\n🤖 [IA MARIA]: Este seria o meu feedback da semana:")
print(json.dumps(maria_feedback, indent=2, ensure_ascii=False))

# (Neste momento ainda não pedimos confirmação à Maria)

# 4. IA RELACIONAL analisa os dois feedbacks
final_report = ai_relational.generate_feedback(rui_feedback, maria_feedback)

# 5. Guardar tudo
save_memory('ai_memory/rui_memory.json', ai_rui.memory)
save_memory('ai_memory/maria_memory.json', ai_maria.memory)
save_memory('ai_memory/relational_memory.json', ai_relational.memory)

save_report(final_report)

print("\n📊 RELATÓRIO FINAL GERADO COM SUCESSO:")
print(json.dumps(final_report, indent=2, ensure_ascii=False))
print("\n✅ Memórias atualizadas e relatório salvo.")
print("🔄 Fim do ciclo semanal de feedback entre IA RUI e IA MARIA.")