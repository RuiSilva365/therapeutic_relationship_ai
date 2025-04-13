import json
import os
from datetime import datetime
from ai.ai_rui import RuiAI
from ai.ai_maria import MariaAI
from ai.ai_relational import RelationalAI
import threading
import pickle
import time

# === UTILS ===
def load_memory(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"⚠️ Erro ao parsear {path}, retornando None")
                return None
    return None

def save_memory(path, ai_memory):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(ai_memory, f, indent=2, ensure_ascii=False)
    print(f"💾 Memória salva em {path}")

def load_conversations(directory):
    merged_data = {"participants": [], "messages": []}
    name_mapping = {"Maria Passos": "Maria", "Rui Silva": "Rui"}
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {directory}")
    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for participant in data.get("participants", []):
                    name = participant.get("name")
                    normalized_name = name_mapping.get(name, name)
                    if not any(p["name"] == normalized_name for p in merged_data["participants"]):
                        merged_data["participants"].append({"name": normalized_name})
                for message in data.get("messages", []):
                    normalized_message = {
                        "sender_name": name_mapping.get(message.get("sender_name"), message.get("sender_name")),
                        "timestamp_ms": message.get("timestamp_ms"),
                        "content": message.get("content", "[Audio message]" if message.get("audio_files") else ""),
                        "reactions": message.get("reactions", [])
                    }
                    merged_data["messages"].append(normalized_message)
        except json.JSONDecodeError:
            print(f"⚠️ Erro ao parsear {file_path}, pulando arquivo")
            continue
    merged_data["messages"].sort(key=lambda x: x["timestamp_ms"])
    return merged_data

def save_report(report):
    today = datetime.today().strftime('%Y-%m-%d')
    path = f'reports/report_{today}.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"📊 Relatório salvo em {path}")

def create_interaction_blocks(messages: list):
    blocks = []
    for i in range(len(messages) - 1):
        input_msg = messages[i]
        response_msg = messages[i + 1]
        if input_msg["sender_name"] != response_msg["sender_name"]:
            blocks.append({
                "input": {
                    "sender": input_msg["sender_name"],
                    "timestamp_ms": input_msg["timestamp_ms"],
                    "message": input_msg.get("content", "")
                },
                "response": {
                    "sender": response_msg["sender_name"],
                    "timestamp_ms": response_msg["timestamp_ms"],
                    "message": response_msg.get("content", "")
                }
            })
    return blocks

def split_into_batches(items, max_tokens_per_batch=2000, is_blocks=False):
    batches = []
    current_batch = []
    current_tokens = 0
    words_per_token = 0.75

    def estimate_tokens(text):
        return int(len(text.split()) / words_per_token)

    for item in items:
        if is_blocks:
            item_text = f"[{datetime.fromtimestamp(item['input']['timestamp_ms'] / 1000).strftime('%Y-%m-%d %H:%M:%S')}] {item['input']['sender']}: {item['input']['message']}\n" + \
                        f"[{datetime.fromtimestamp(item['response']['timestamp_ms'] / 1000).strftime('%Y-%m-%d %H:%M:%S')}] {item['response']['sender']}: {item['response']['message']}"
        else:
            item_text = "\n".join(f"[{datetime.fromtimestamp(msg['timestamp_ms'] / 1000).strftime('%Y-%m-%d %H:%M:%S')}] {msg['sender_name']}: {msg['content']}" for msg in item)
        token_count = estimate_tokens(item_text)
        if current_tokens + token_count > max_tokens_per_batch:
            if current_batch:
                batches.append(current_batch)
            current_batch = [item]
            current_tokens = token_count
        else:
            current_batch.append(item)
            current_tokens += token_count
    if current_batch:
        batches.append(current_batch)
    return batches

def load_cache(cache_path):
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError):
            print(f"⚠️ Erro ao carregar cache {cache_path}, retornando vazio")
            return {}
    return {}

def save_cache(cache_path, cache_data):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"💾 Cache salvo em {cache_path}")
    except Exception as e:
        print(f"❌ Erro ao salvar cache {cache_path}: {str(e)}")

# === INÍCIO DO SCRIPT ===
MODEL_URL = "http://192.168.56.1:1234"
rui_cache_path = 'data/rui_cache.pkl'
maria_cache_path = 'data/maria_cache.pkl'

# Carregar memórias e caches
rui_memory = load_memory('data/rui_memory')
maria_memory = load_memory('data/maria_memory')
relational_memory = load_memory('data/relational_memory.json') or {"dinamicas_relacionais": []}
rui_cache = load_cache(rui_cache_path)
maria_cache = load_cache(maria_cache_path)

# Inicializar AIs
ai_rui = RuiAI(memory=rui_memory, model_url=MODEL_URL)
ai_maria = MariaAI(memory=maria_memory, model_url=MODEL_URL)
ai_relational = RelationalAI(memory=relational_memory, model_url=MODEL_URL)

# Carregar conversas
try:
    conversation_data = load_conversations('data')
    messages = conversation_data.get("messages", [])
    print(f"🔍 Total de mensagens: {len(messages)}")
except FileNotFoundError as e:
    print(f"❌ Erro: {str(e)}")
    exit(1)

# Criar blocos de interação
blocks = create_interaction_blocks(messages)
print(f"🔄 Blocos de interação: {len(blocks)}")
if blocks:
    print(f"📋 Amostra do bloco 1: {json.dumps(blocks[0], ensure_ascii=False)}")
else:
    print("⚠️ Nenhum bloco de interação gerado, encerrando.")
    exit(1)

# Gerar memórias iniciais, se necessário
if rui_memory is None:
    print("📝 Gerando memória inicial para Rui...")
    ai_rui.generate_initial_memory(blocks)
    save_memory('data/rui_memory', ai_rui.memory)
if maria_memory is None:
    print("📝 Gerando memória inicial para Maria...")
    ai_maria.generate_initial_memory(blocks)
    save_memory('data/maria_memory', ai_maria.memory)

# Dividir blocos em lotes
batches = split_into_batches(blocks, max_tokens_per_batch=3000, is_blocks=True)
print(f"📝 Processando {len(batches)} lotes...")

# Inicializar estruturas de feedback
rui_feedback = {
    "reflexoes": [],
    "planos": [],
    "elogios": [],
    "core_values": [],
    "emotional_patterns": []
}
maria_feedback = {
    "reflexoes": [],
    "planupakan": [],
    "planos": [],
    "elogios": [],
    "core_values": [],
    "emotional_patterns": []
}

total_start_time = time.time()
lock = threading.Lock()

for i, batch in enumerate(batches, 1):
    batch_key = f"batch_{i}"
    print(f"\n=== Lote {i}/{len(batches)} ===")
    start_time = time.time()

    def process_rui():
        with lock:
            if batch_key in rui_cache:
                print(f"📥 Cache hit para Rui, lote {i}")
                return rui_cache[batch_key]
            feedback = ai_rui.analyze(batch)
            if any(feedback.get(key) for key in rui_feedback):
                rui_cache[batch_key] = feedback
            return feedback

    def process_maria():
        with lock:
            if batch_key in maria_cache:
                print(f"📥 Cache hit para Maria, lote {i}")
                return maria_cache[batch_key]
            feedback = ai_maria.analyze(batch)
            if any(feedback.get(key) for key in maria_feedback):
                maria_cache[batch_key] = feedback
            return feedback

    rui_thread = threading.Thread(target=lambda: rui_feedback.update(process_rui()))
    maria_thread = threading.Thread(target=lambda: maria_feedback.update(process_maria()))
    rui_thread.start()
    maria_thread.start()
    rui_thread.join()
    maria_thread.join()

    # Acumular feedback
    rui_batch_feedback = rui_feedback.copy()
    maria_batch_feedback = maria_feedback.copy()

    for key in ["reflexoes", "planos", "elogios", "core_values", "emotional_patterns"]:
        if key in rui_batch_feedback:
            rui_feedback[key].extend(rui_batch_feedback[key])
        if key in maria_batch_feedback:
            maria_feedback[key].extend(maria_batch_feedback[key])

    print(f"🤖 [Rui] Lote {i}: {json.dumps(rui_batch_feedback, indent=2, ensure_ascii=False)[:200]}...")
    print(f"🤖 [Maria] Lote {i}: {json.dumps(maria_batch_feedback, indent=2, ensure_ascii=False)[:200]}...")
    print(f"⏱ Tempo do lote {i}: {time.time() - start_time:.2f} segundos")

# Salvar caches
save_cache(rui_cache_path, rui_cache)
save_cache(maria_cache_path, maria_cache)

# Gerar relatório final
final_report = ai_relational.generate_feedback(rui_feedback, maria_feedback)
print("\n❤️ Relatório final:", json.dumps(final_report, indent=2, ensure_ascii=False))

# Salvar memórias e relatório
save_memory('data/rui_memory', ai_rui.memory)
save_memory('data/maria_memory', ai_maria.memory)
save_memory('data/relational_memory.json', ai_relational.memory)
save_report(final_report)

print("\n=== RESUMO FINAL ===")
print(f"🧠 Rui: {json.dumps(rui_feedback, indent=2, ensure_ascii=False)[:200]}...")
print(f"🧠 Maria: {json.dumps(maria_feedback, indent=2, ensure_ascii=False)[:200]}...")
print(f"❤️ Relacional: {json.dumps(final_report, indent=2, ensure_ascii=False)[:200]}...")
print(f"⏱ Tempo total: {(time.time() - total_start_time) / 60:.2f} minutos")