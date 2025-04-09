import json
import requests

LM_API_URL = "http://localhost:1234/v1/completions"
MODEL_NAME = "openhermes-2.5-mistral-7b"

def load_conversations(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("messages", [])

def format_conversation(messages):
    formatted = ""
    for msg in messages:
        sender = msg.get("sender_name", "Desconhecido")
        content = msg.get("content", "")
        formatted += f"{sender}: {content}\n"
    return formatted.strip()

def generate_prompt(conversation_text):
    return f"""
Tu és um analista emocional muito sensível e especializado em relações amorosas.

Com base nesta conversa entre duas pessoas (Rui e Maria), gera um relatório emocional com 3 seções:
1. 🧠 Reflexões emocionais do Rui (inseguranças, stress, sentimentos expressos)
2. 💬 Intenções e planos futuros (desejos, propostas de atividades, pedidos)
3. ❤️ Elogios e gestos de carinho entre ambos

Tenta detetar ironia, preocupação genuína e diferenças subtis de tom. Escreve em português de forma natural, como um psicólogo que faz um resumo semanal.

Conversa:
{conversation_text}

Relatório emocional:
"""

def ask_model(prompt):
    payload = {
        "prompt": prompt,
        "max_tokens": 800,
        "temperature": 0.7,
        "stop": ["</s>"],
        "model": MODEL_NAME
    }

    response = requests.post(LM_API_URL, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["text"]

def main():
    messages = load_conversations("data/conversation_simulated.json")
    if not messages:
        print("❌ Nenhuma mensagem encontrada.")
        return

    print(f"🔍 Mensagens carregadas: {len(messages)}")

    # Corrigir caso haja um nested list
    if isinstance(messages[0], list):
        messages = messages[0]

    conversation_text = format_conversation(messages)
    prompt = generate_prompt(conversation_text)
    
    print("🤖 A pensar...")
    resposta = ask_model(prompt)
    print("\n📋 RELATÓRIO EMOCIONAL GERADO:\n")
    print(resposta.strip())

if __name__ == "__main__":
    main()
