import json
import os
from datetime import datetime
from ai.ai_rui import RuiAI
from ai.ai_maria import MariaAI
from ai.ai_relational import RelationalAI
import time

# === UTILS ===
def load_memory(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            try:
                data = json.loads(content)
                print(f"📂 Carregado {path}: {json.dumps(data, ensure_ascii=False)[:200]}...")
                if path.endswith('relational_memory.json'):
                    if "relational_dynamics" in data and not isinstance(data["relational_dynamics"], list):
                        print(f"⚠️ 'relational_dynamics' inválido em {path}, corrigindo para lista")
                        data["relational_dynamics"] = []
                return data
            except json.JSONDecodeError:
                print(f"⚠️ Erro ao parsear {path}, retornando vazio")
                return {}
    print(f"📂 {path} não existe, retornando vazio")
    return {}

def save_memory(path, ai_memory):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(ai_memory, f, indent=2, ensure_ascii=False)
    print(f"💾 Memória salva em {path}")

def save_report(report):
    today = datetime.today().strftime('%Y-%m-%d')
    path = f'reports/report_{today}.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"📊 Relatório salvo em {path}")

def load_conversations(directory):
    merged_data = {"participants": [], "messages": []}
    name_mapping = {"Maria Passos": "Maria", "Rui Silva": "Rui"}
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"Nenhum arquivo JSON encontrado em {directory}")
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
                        "content": message.get("content", "[Mensagem de áudio]" if message.get("audio_files") else ""),
                        "reactions": message.get("reactions", [])
                    }
                    merged_data["messages"].append(normalized_message)
        except json.JSONDecodeError:
            print(f"⚠️ Erro ao parsear {file_path}, pulando arquivo")
            continue
    merged_data["messages"].sort(key=lambda x: x["timestamp_ms"])
    return merged_data

def create_interaction_blocks(messages: list, max_blocks: int = None):
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
    if max_blocks is not None:
        print(f"📏 Total de blocos gerados: {len(blocks)}, limitando a {max_blocks}")
        return blocks[:max_blocks]
    print(f"📏 Total de blocos gerados: {len(blocks)}")
    return blocks

# === INÍCIO DO SCRIPT ===
MODEL_URL = "http://192.168.56.1:1234"

def main():
    # Carregar memórias
    rui_memory = load_memory('data/rui_memory.json')
    maria_memory = load_memory('data/maria_memory.json')
    relational_memory = load_memory('data/relational_memory.json')
    if not relational_memory:
        relational_memory = {
            "rui_profile": {},
            "maria_profile": {},
            "relational_dynamics": []
        }
        print("📂 Inicializando relational_memory padrão")

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
        return

    # Gerar memórias iniciais, se necessário
    if not rui_memory or rui_memory == ai_rui.MEMORY_SCHEMA:
        print("📝 Gerando perfil inicial para Rui...")
        all_blocks = create_interaction_blocks(messages)
        if not all_blocks:
            print("❌ Nenhuma interação válida para gerar perfil de Rui.")
            return
        ai_rui.generate_initial_memory(all_blocks)
        if ai_rui.memory == ai_rui.MEMORY_SCHEMA:
            print("❌ Falha ao gerar perfil para Rui. Verifique a API.")
            return
        save_memory('data/rui_memory.json', ai_rui.memory)
    if not maria_memory or maria_memory == ai_maria.MEMORY_SCHEMA:
        print("📝 Gerando perfil inicial para Maria...")
        all_blocks = create_interaction_blocks(messages)
        if not all_blocks:
            print("❌ Nenhuma interação válida para gerar perfil de Maria.")
            return
        ai_maria.generate_initial_memory(all_blocks)
        if ai_maria.memory == ai_maria.MEMORY_SCHEMA:
            print("❌ Falha ao gerar perfil para Maria. Verifique a API.")
            return
        save_memory('data/maria_memory.json', ai_maria.memory)

    # Usar toda a conversa para análise
    recent_blocks = create_interaction_blocks(messages)
    print(f"🔄 Blocos recentes: {len(recent_blocks)}")
    if recent_blocks:
        print(f"📋 Amostra do bloco recente 1: {json.dumps(recent_blocks[0], ensure_ascii=False)}")
        print(f"📋 Amostra de blocos recentes: {json.dumps(recent_blocks[:3], ensure_ascii=False)[:500]}...")

    # Analisar mensagens
    rui_feedback = ai_rui.analyze(recent_blocks)
    maria_feedback = ai_maria.analyze(recent_blocks)
    print(f"📜 Feedback Rui: {json.dumps(rui_feedback, ensure_ascii=False)[:200]}...")
    print(f"📜 Feedback Maria: {json.dumps(maria_feedback, ensure_ascii=False)[:200]}...")
    if not rui_feedback.get("recent_reflections"):
        print("⚠️ Nenhuma reflexão para Rui.")
    if not maria_feedback.get("recent_reflections"):
        print("⚠️ Nenhuma reflexão para Maria.")
    if not (rui_feedback.get("recent_reflections") and maria_feedback.get("recent_reflections")):
        print("⚠️ Reflexões incompletas, prosseguindo com feedback disponível.")

    # Gerar relatório relacional
    final_report = ai_relational.generate_feedback(rui_feedback, maria_feedback)
    print(f"📜 Relatório final: {json.dumps(final_report, ensure_ascii=False)[:200]}...")
    if not (final_report.get("strengths") or final_report.get("challenges") or final_report.get("advice")):
        print("❌ Relatório relacional vazio. Verifique a API.")
        return

    # Salvar memórias e relatório
    save_memory('data/rui_memory.json', ai_rui.memory)
    save_memory('data/maria_memory.json', ai_maria.memory)
    save_memory('data/relational_memory.json', ai_relational.memory)
    save_report(final_report)

    # Exibir resumo
    print("\n=== RESUMO FINAL ===")
    print(f"🧠 Rui: {json.dumps(rui_feedback, indent=2, ensure_ascii=False)[:200]}...")
    print(f"🧠 Maria: {json.dumps(maria_feedback, indent=2, ensure_ascii=False)[:200]}...")
    print(f"❤️ Relacional: {json.dumps(final_report, indent=2, ensure_ascii=False)[:200]}...")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"⏱ Tempo total: {(time.time() - start_time) / 60:.2f} minutos")