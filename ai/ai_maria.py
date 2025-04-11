from datetime import datetime
import json
import requests
import math
import re

class MariaAI:
    def __init__(self, memory: dict, model_url: str):
        self.memory = memory or {
            "personalidade": "Indefinida", 
            "valores": [], 
            "gatilhos_emocionais": [], 
            "reflexoes": [], 
            "planos": [], 
            "elogios": [],
            "relationship_metrics": []  # New field to track relationship metrics over time
        }
        self.model_url = model_url
        self.MAX_TOKENS_PER_BATCH = 2000
        
    def _generate_conversation_summary(self, conversation_turns: list) -> str:
        """Generate a brief summary of the conversation to maintain context between batches."""
        conversation_text = self.format_conversation(conversation_turns[:5])  # Use first 5 turns for summary
        prompt = f"""
        Resume esta conversa entre Rui e Maria em 3-5 frases. Foca nos temas principais, 
        tom emocional, e pontos importantes mencionados:
        
        {conversation_text}
        """
        response = self.ask_model(prompt)
        summary_text = response.get("choices", [{}])[0].get("text", "").strip()
        return summary_text

    def generate_initial_memory(self, conversation_turns: list):
        # Generate a summary for better context
        conversation_summary = self._generate_conversation_summary(conversation_turns)
        print(f"📝 Resumo da conversa: {conversation_summary}")
        
        # Dividir turnos em lotes
        batches = self._split_into_batches(conversation_turns)
        print(f"📝 Gerando memória inicial para Maria em {len(batches)} lotes.")

        accumulated_profile = {
            "personalidade": "Indefinida",
            "valores": [],
            "gatilhos_emocionais": [],
            "reflexoes": [],
            "planos": [],
            "elogios": []
        }

        previous_feedback = ""
        for i, batch in enumerate(batches):
            conversation_text = self.format_conversation(batch)
            prompt = f"""
            Analisa APENAS esta parte da conversa entre Rui e Maria (lote {i+1}/{len(batches)}) e atualiza o perfil psicológico da Maria.
            
            Resumo geral da conversa: {conversation_summary}
            
            Inclui:
            - personalidade (ex.: empática, extrovertida)
            - valores (ex.: compreensão, diálogo)
            - gatilhos emocionais (ex.: rejeição, ignorada)
            
            Retorna SOMENTE o JSON com essas chaves no nível raiz, sem texto adicional ou aninhamento.
            Exemplo: {{"personalidade": "empática", "valores": ["compreensão"], "gatilhos_emocionais": ["rejeição"]}}

            Feedback anterior (para contexto, se aplicável):
            {previous_feedback}

            Conversa:
            {conversation_text}
            """
            print(f"📋 Prompt para lote {i+1}: {prompt[:200]}...")
            print(f"📏 Tamanho estimado do prompt: ~{len(prompt.split()) // 0.75} tokens")
            
            try:
                response = self.ask_model(prompt)
                profile_text = response.get("choices", [{}])[0].get("text", "{}").strip()

                # Extrair JSON
                json_match = re.search(r'\{.*\}', profile_text, re.DOTALL)
                if json_match:
                    profile_json = json_match.group(0)
                    try:
                        profile_data = json.loads(profile_json)
                        # Atualizar perfil acumulado com novos dados
                        if profile_data.get("personalidade") and profile_data["personalidade"] != "Indefinida":
                            accumulated_profile["personalidade"] = profile_data["personalidade"]
                        accumulated_profile["valores"] = list(set(accumulated_profile["valores"] + profile_data.get("valores", [])))
                        accumulated_profile["gatilhos_emocionais"] = list(set(accumulated_profile["gatilhos_emocionais"] + profile_data.get("gatilhos_emocionais", [])))
                        previous_feedback = json.dumps(profile_data, ensure_ascii=False)
                    except json.JSONDecodeError:
                        print(f"Erro ao parsear JSON do perfil de Maria (lote {i+1}): {profile_json}")
                else:
                    print(f"Nenhum JSON válido encontrado para Maria (lote {i+1}): {profile_text}")
            except Exception as e:
                print(f"Erro ao processar lote {i+1}: {str(e)}")

        # Finalizar memória
        self.memory.update(accumulated_profile)
        self.memory.setdefault("reflexoes", [])
        self.memory.setdefault("planos", [])
        self.memory.setdefault("elogios", [])
        self.memory.setdefault("relationship_metrics", [])
        print(f"Memória inicial gerada para Maria: {json.dumps(self.memory, indent=2, ensure_ascii=False)}")

    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment of text using the model."""
        prompt = f"""
        Analise o sentimento do seguinte texto numa escala de -5 (muito negativo) 
        a +5 (muito positivo). Identifique também emoções presentes (alegria, tristeza, 
        raiva, medo, surpresa, etc). Responda apenas com um JSON:
        
        Texto: {text}
        """
        response = self.ask_model(prompt)
        sentiment_text = response.get("choices", [{}])[0].get("text", "{}")
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', sentiment_text, re.DOTALL)
        if json_match:
            sentiment_json = json_match.group(0)
            try:
                return json.loads(sentiment_json)
            except:
                return {"score": 0, "emotions": []}
        return {"score": 0, "emotions": []}

    def analyze(self, conversation_turns: list) -> dict:
        # Generate a summary for better context
        conversation_summary = self._generate_conversation_summary(conversation_turns)
        print(f"📝 Resumo da conversa para análise: {conversation_summary}")
        
        # Dividir turnos em lotes
        batches = self._split_into_batches(conversation_turns)
        print(f"📝 Analisando conversa para Maria em {len(batches)} lotes.")

        accumulated_feedback = {
            "reflexoes": [],
            "planos": [],
            "elogios": [],
            "valores": self.memory.get("valores", []),
            "gatilhos_emocionais": self.memory.get("gatilhos_emocionais", []),
            "sentiment_scores": []  # New field to track sentiment across the conversation
        }
        previous_feedback = ""

        for i, batch in enumerate(batches):
            conversation_text = self.format_conversation(batch)
            
            # Analyze sentiment for this batch
            sentiment = self.analyze_sentiment(conversation_text)
            accumulated_feedback["sentiment_scores"].append({
                "batch": i+1,
                "score": sentiment.get("score", 0),
                "emotions": sentiment.get("emotions", [])
            })
            
            prompt = self.generate_prompt(
                conversation_text, 
                previous_feedback, 
                batch_number=i+1, 
                total_batches=len(batches),
                conversation_summary=conversation_summary,
                sentiment=sentiment
            )
            
            try:
                feedback = self.ask_model(prompt)
                feedback_text = feedback.get("choices", [{}])[0].get("text", "").strip()
                reflexoes, planos, elogios = [], [], []
                
                # Process the lines to extract reflections, plans, and compliments
                for line in feedback_text.split("\n"):
                    line = line.strip()
                    if line:
                        if "Reflexões emocionais" in line or "sinto" in line.lower():
                            reflexoes.append({"data": datetime.now().strftime("%Y-%m-%d"), "texto": line})
                        elif "Planos futuros" in line or "gostava de" in line.lower():
                            planos.append({"data": datetime.now().strftime("%Y-%m-%d"), "texto": line})
                        elif "Gestos de carinho" in line or "disse" in line.lower():
                            elogios.append({"data": datetime.now().strftime("%Y-%m-%d"), "texto": line})

                accumulated_feedback["reflexoes"].extend(reflexoes)
                accumulated_feedback["planos"].extend(planos)
                accumulated_feedback["elogios"].extend(elogios)
                
                # Update values and emotional triggers based on keywords in the feedback
                feedback_lower = feedback_text.lower()
                for value in ["compreensão", "diálogo", "comunicação", "honestidade", "apoio"]:
                    if value in feedback_lower and value not in accumulated_feedback["valores"]:
                        accumulated_feedback["valores"].append(value)
                
                for trigger in ["rejeição", "ignorada", "criticada", "abandonada", "desvalorizada"]:
                    if trigger in feedback_lower and trigger not in accumulated_feedback["gatilhos_emocionais"]:
                        accumulated_feedback["gatilhos_emocionais"].append(trigger)
                
                previous_feedback = feedback_text
                print(f"Feedback parcial Maria (lote {i+1}): {feedback_text[:200]}...")
            except Exception as e:
                print(f"Erro ao processar lote {i+1}: {str(e)}")

        # Update relationship metrics
        self.update_relationship_metrics(accumulated_feedback)
        
        # Atualizar memória
        self.memory["reflexoes"].extend(accumulated_feedback["reflexoes"])
        self.memory["planos"].extend(accumulated_feedback["planos"])
        self.memory["elogios"].extend(accumulated_feedback["elogios"])
        self.memory["valores"] = list(set(self.memory["valores"] + accumulated_feedback["valores"]))
        self.memory["gatilhos_emocionais"] = list(set(self.memory["gatilhos_emocionais"] + accumulated_feedback["gatilhos_emocionais"]))
        
        return accumulated_feedback

    def update_relationship_metrics(self, feedback: dict) -> None:
        """Track relationship metrics over time."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate average sentiment score
        if feedback.get("sentiment_scores"):
            avg_sentiment = sum(item["score"] for item in feedback["sentiment_scores"]) / len(feedback["sentiment_scores"])
        else:
            avg_sentiment = 0
            
        # Count negative emotional triggers mentioned
        trigger_count = len([r for r in feedback.get("reflexoes", []) 
                            if any(t in r["texto"].lower() for t in self.memory.get("gatilhos_emocionais", []))])
        
        # Count positive interactions
        positive_count = len(feedback.get("elogios", []))
        
        metrics = {
            "date": today,
            "sentiment_score": avg_sentiment,
            "emotional_triggers": trigger_count,
            "positive_interactions": positive_count,
            "emotional_balance": avg_sentiment - (trigger_count * 0.5) + (positive_count * 0.5)
        }
        
        self.memory.setdefault("relationship_metrics", []).append(metrics)
        print(f"Métricas de relacionamento atualizadas: {metrics}")

    def format_conversation(self, turns: list) -> str:
        formatted = ""
        for turn in turns:
            for msg in turn:
                sender = msg.get("sender_name", "Desconhecido")
                content = msg.get("content", "")
                timestamp = datetime.fromtimestamp(msg.get("timestamp_ms", 0) / 1000).strftime("%Y-%m-%d %H:%M:%S")
                reactions = msg.get("reactions", [])
                reaction_str = f" [reagiu com {', '.join(r['reaction'] for r in reactions)}]" if reactions else ""
                formatted += f"[{timestamp}] {sender}: {content}{reaction_str}\n"
        return formatted.strip()

    def generate_prompt(self, conversation_text: str, previous_feedback: str, batch_number: int, 
                       total_batches: int, conversation_summary: str = "", sentiment: dict = None) -> str:
        personalidade = self.memory.get("personalidade", "Indefinida")
        valores = ", ".join(self.memory.get("valores", []))
        gatilhos = ", ".join(self.memory.get("gatilhos_emocionais", []))
        
        sentiment_info = ""
        if sentiment:
            sentiment_info = f"""
            Análise de sentimento desta parte da conversa:
            - Pontuação: {sentiment.get('score', 0)} (de -5 a +5)
            - Emoções detectadas: {', '.join(sentiment.get('emotions', []))}
            """
            
        return f"""
        Tu és a Maria. A tua personalidade é {personalidade}. Valorizas {valores}. Os teus gatilhos emocionais incluem {gatilhos}.
        
        Resumo geral da conversa: {conversation_summary}
        
        {sentiment_info}
        
        A partir desta parte da conversa (lote {batch_number}/{total_batches}), gera um relatório emocional focando-te na Maria.
        Sê específica e inclui:
        - Reflexões emocionais: como me sinto em relação ao que foi dito e porquê
        - Planos futuros: o que pretendo fazer ou mudar
        - Gestos de carinho: momentos em que mostrei ou recebi afeto
        
        Fala na primeira pessoa e usa exemplos específicos da conversa.
        
        Considera o feedback anterior para manter consistência:
        {previous_feedback}
        
        Conversa:
        {conversation_text}
        
        Relatório emocional:
        """

    def _split_into_batches(self, turns: list) -> list:
        """Divide os turnos em lotes com base no limite de tokens, mantendo contexto."""
        batches = []
        current_batch = []
        current_tokens = 0
        words_per_token = 0.75  # Estimativa de tokens por palavra

        for turn in turns:
            turn_text = self.format_conversation([turn])
            word_count = len(turn_text.split())
            token_count = math.ceil(word_count / words_per_token)

            # Verificar se este turno cabe no lote atual
            if current_tokens + token_count > self.MAX_TOKENS_PER_BATCH:
                if current_batch:
                    batches.append(current_batch)
                # Iniciar novo lote com sobreposição para manter contexto
                if len(current_batch) > 1:
                    current_batch = current_batch[-1:] + [turn]  # Manter último turno do lote anterior
                    current_tokens = token_count + math.ceil(len(self.format_conversation([current_batch[0]]).split()) / words_per_token)
                else:
                    current_batch = [turn]
                    current_tokens = token_count
            else:
                current_batch.append(turn)
                current_tokens += token_count

        if current_batch:
            batches.append(current_batch)

        return batches

    def ask_model(self, prompt: str) -> dict:
        headers = {'Content-Type': 'application/json'}
        data = {'model': 'openhermes-2.5-mistral-7b', 'prompt': prompt, 'max_tokens': 1000}
        try:
            response = requests.post(f"{self.model_url}/v1/completions", headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"❌ Erro ao chamar a API do modelo: {str(e)}")
            raise Exception(f"Erro ao chamar a API do modelo: {str(e)}")

    def generate_actionable_recommendations(self) -> list:
        """Generate specific, actionable recommendations based on memory."""
        personalidade = self.memory.get("personalidade", "Indefinida")
        valores = self.memory.get("valores", [])
        gatilhos = self.memory.get("gatilhos_emocionais", [])
        reflexoes = [r["texto"] for r in self.memory.get("reflexoes", [])[-5:]]  # Last 5 reflections
        planos = [p["texto"] for p in self.memory.get("planos", [])[-5:]]  # Last 5 plans
        
        prompt = f"""
        Com base no perfil psicológico da Maria:
        - Personalidade: {personalidade}
        - Valores: {', '.join(valores)}
        - Gatilhos emocionais: {', '.join(gatilhos)}
        - Reflexões recentes: {'; '.join(reflexoes)}
        - Planos recentes: {'; '.join(planos)}
        
        Gere 3 recomendações ESPECÍFICAS e ACIONÁVEIS para melhorar o relacionamento da Maria com o Rui.
        Cada recomendação deve incluir:
        1. O problema ou oportunidade específica
        2. Uma ação concreta para implementar esta semana
        3. Como avaliar se a ação teve efeito positivo
        
        Responda em JSON com um array de objetos, cada um contendo "problema", "acao", e "avaliacao".
        """
        
        response = self.ask_model(prompt)
        json_text = response.get("choices", [{}])[0].get("text", "{}").strip()
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
        if json_match:
            try:
                recommendations = json.loads(json_match.group(0))
                return recommendations.get("recomendacoes", [])
            except:
                return []
        return []