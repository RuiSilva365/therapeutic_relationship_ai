from datetime import datetime
import json
import requests
import math
import re

class RuiAI:
    def __init__(self, memory: dict, model_url: str):
        self.memory = memory or {
            "personalidade": "Indefinida", 
            "valores": [], 
            "gatilhos_emocionais": [], 
            "reflexoes": [], 
            "planos": [], 
            "elogios": [],
            "relationship_metrics": []
        }
        self.model_url = model_url
        self.MAX_TOKENS_PER_BATCH = 1500  # Increased for better context

    def _generate_conversation_summary(self, conversation_turns: list) -> str:
        """Generate a summary of the conversation to maintain context between batches."""
        if not conversation_turns:
            return "Sem conversa para resumir."
            
        # Use a sample of turns for summary (first, middle and last)
        sample_turns = []
        if len(conversation_turns) > 0:
            sample_turns.append(conversation_turns[0])  # First turn
        if len(conversation_turns) > 2:
            sample_turns.append(conversation_turns[len(conversation_turns) // 2])  # Middle turn
        if len(conversation_turns) > 1:
            sample_turns.append(conversation_turns[-1])  # Last turn
            
        conversation_text = self.format_conversation(sample_turns)
        prompt = f"""
        Resume brevemente esta conversa entre Rui e Maria em 3-5 frases. 
        Foca nos temas principais e no tom emocional da conversa:
        
        {conversation_text}
        """
        try:
            response = self.ask_model(prompt)
            summary_text = response.get("choices", [{}])[0].get("text", "").strip()
            return summary_text
        except Exception as e:
            print(f"❌ Erro ao gerar resumo: {str(e)}")
            return "Falha ao gerar resumo da conversa."

    def generate_initial_memory(self, conversation_turns: list):
        # Generate summary for better context
        conversation_summary = self._generate_conversation_summary(conversation_turns)
        print(f"📝 Resumo da conversa: {conversation_summary}")
        
        batches = self._split_into_batches(conversation_turns)
        print(f"📝 Gerando memória inicial para Rui em {len(batches)} lotes.")

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
            Analisa ESTA PARTE da conversa entre Rui e Maria (lote {i+1}/{len(batches)}) e atualiza o perfil psicológico do Rui com base apenas nas mensagens fornecidas.
            
            Resumo geral da conversa: {conversation_summary}
            
            Inclui:
            - personalidade: descreve traços específicos (ex.: introspectivo, pragmático) com base em como Rui fala ou responde.
            - valores: identifica valores implícitos nas mensagens (ex.: conexão, honestidade) com exemplos breves.
            - gatilhos_emocionais: detecta emoções fortes ou reações (ex.: stress, rejeição) e explica o que os causou.
            
            Retorna SOMENTE o JSON com essas chaves no nível raiz, sem texto adicional ou aninhamento.
            Exemplo: {{"personalidade": "introspectivo", "valores": ["conexão"], "gatilhos_emocionais": ["stress"]}}
            Se não houver informação suficiente, mantém os valores anteriores ou usa "Indefinido".

            Feedback anterior (para refinar, se aplicável):
            {previous_feedback}

            Conversa:
            {conversation_text}
            """

            try:
                response = self.ask_model(prompt)
                profile_text = response.get("choices", [{}])[0].get("text", "{}").strip()
                print(f"📤 Texto bruto retornado para Rui (lote {i+1}): {profile_text[:200]}...")

                # Extrair JSON
                json_match = re.search(r'\{.*\}', profile_text, re.DOTALL)
                if json_match:
                    profile_json = json_match.group(0)
                    try:
                        profile_data = json.loads(profile_json)
                        # Atualizar personalidade (priorizar a mais recente, se definida)
                        if profile_data.get("personalidade") and profile_data["personalidade"] != "Indefinido":
                            accumulated_profile["personalidade"] = profile_data["personalidade"]
                        # Acumular valores e gatilhos, normalizando duplicatas
                        accumulated_profile["valores"] = list(set(accumulated_profile["valores"] + profile_data.get("valores", [])))
                        triggers = profile_data.get("gatilhos_emocionais", [])
                        # Normalizar "stress" e "estresse"
                        normalized_triggers = [t.lower().replace("estresse", "stress") for t in triggers]
                        accumulated_profile["gatilhos_emocionais"] = list(set(accumulated_profile["gatilhos_emocionais"] + normalized_triggers))
                        previous_feedback = json.dumps(profile_data, ensure_ascii=False)  # Passar como JSON limpo
                    except json.JSONDecodeError:
                        print(f"❌ Erro ao parsear JSON do perfil de Rui (lote {i+1}): {profile_json}")
                else:
                    print(f"❌ Nenhum JSON válido encontrado para Rui (lote {i+1}): {profile_text}")
            except Exception as e:
                print(f"❌ Erro ao processar lote {i+1}: {str(e)}")
                continue

        self.memory.update(accumulated_profile)
        self.memory.setdefault("reflexoes", [])
        self.memory.setdefault("planos", [])
        self.memory.setdefault("elogios", [])
        self.memory.setdefault("relationship_metrics", [])
        print(f"✅ Memória inicial gerada para Rui: {json.dumps(self.memory, indent=2, ensure_ascii=False)}")

    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment and emotions in the text."""
        prompt = f"""
        Analisa o sentimento do seguinte texto numa escala de -5 (muito negativo) a +5 (muito positivo).
        Identifica também as emoções presentes (alegria, tristeza, raiva, medo, surpresa, etc).
        Responde apenas com um JSON contendo "score" (número) e "emotions" (array de strings):
        
        Texto: {text}
        """
        
        try:
            response = self.ask_model(prompt)
            sentiment_text = response.get("choices", [{}])[0].get("text", "{}").strip()
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', sentiment_text, re.DOTALL)
            if json_match:
                sentiment_json = json_match.group(0)
                try:
                    return json.loads(sentiment_json)
                except json.JSONDecodeError:
                    print(f"❌ Erro ao parsear JSON do sentimento: {sentiment_json}")
            return {"score": 0, "emotions": []}
        except Exception as e:
            print(f"❌ Erro ao analisar sentimento: {str(e)}")
            return {"score": 0, "emotions": []}

    def analyze(self, conversation_turns: list) -> dict:
        # Generate summary for context
        conversation_summary = self._generate_conversation_summary(conversation_turns)
        print(f"📝 Resumo da conversa para análise: {conversation_summary}")
        
        batches = self._split_into_batches(conversation_turns)
        print(f"📝 Analisando conversa para Rui em {len(batches)} lotes.")

        accumulated_feedback = {
            "reflexoes": [],
            "planos": [],
            "elogios": [],
            "valores": self.memory.get("valores", []),
            "gatilhos_emocionais": self.memory.get("gatilhos_emocionais", []),
            "sentiment_analysis": []
        }
        previous_feedback = ""

        for i, batch in enumerate(batches):
            conversation_text = self.format_conversation(batch)
            
            # Analyze sentiment for this batch
            sentiment = self.analyze_sentiment(conversation_text)
            accumulated_feedback["sentiment_analysis"].append({
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
                
                # Process feedback text into categories
                current_category = None
                for line in feedback_text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                        
                    if "Reflexões emocionais" in line:
                        current_category = "reflexoes"
                        continue
                    elif "Planos futuros" in line:
                        current_category = "planos"
                        continue
                    elif "Gestos de carinho" in line:
                        current_category = "elogios"
                        continue
                    
                    # Categorize based on keywords if no explicit category
                    if current_category is None:
                        if "sinto" in line.lower() or "sentir" in line.lower() or "emocion" in line.lower():
                            current_category = "reflexoes"
                        elif "plano" in line.lower() or "gostava de" in line.lower() or "pretendo" in line.lower():
                            current_category = "planos"
                        elif "carinho" in line.lower() or "disse" in line.lower() or "elogio" in line.lower():
                            current_category = "elogios"
                        else:
                            current_category = "reflexoes"  # Default category
                    
                    # Add the line to the appropriate category
                    if current_category == "reflexoes":
                        reflexoes.append({"data": datetime.now().strftime("%Y-%m-%d"), "texto": line})
                    elif current_category == "planos":
                        planos.append({"data": datetime.now().strftime("%Y-%m-%d"), "texto": line})
                    elif current_category == "elogios":
                        elogios.append({"data": datetime.now().strftime("%Y-%m-%d"), "texto": line})

                accumulated_feedback["reflexoes"].extend(reflexoes)
                accumulated_feedback["planos"].extend(planos)
                accumulated_feedback["elogios"].extend(elogios)

                # Update values and triggers based on the feedback
                feedback_lower = feedback_text.lower()
                for value in ["conexão", "honestidade", "transparência", "lealdade", "confiança"]:
                    if value in feedback_lower and value not in accumulated_feedback["valores"]:
                        accumulated_feedback["valores"].append(value)
                
                for trigger in ["stress", "rejeição", "crítica", "frustração", "abandono"]:
                    normalized_trigger = trigger.lower().replace("estresse", "stress")
                    if normalized_trigger in feedback_lower and normalized_trigger not in accumulated_feedback["gatilhos_emocionais"]:
                        accumulated_feedback["gatilhos_emocionais"].append(normalized_trigger)

                previous_feedback = feedback_text
                print(f"📤 Feedback parcial Rui (lote {i+1}): {feedback_text[:200]}...")
            except Exception as e:
                print(f"❌ Erro ao processar lote {i+1} (análise): {str(e)}")
                continue

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
        sentiment_scores = feedback.get("sentiment_analysis", [])
        avg_sentiment = sum(item["score"] for item in sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
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
        print(f"📊 Métricas de relacionamento atualizadas para Rui: {metrics}")

    def format_conversation(self, turns: list) -> str:
        """Format conversation turns into a readable text format."""
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
        """Generate a prompt for analyzing the conversation batch."""
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
        Tu és o Rui. A tua personalidade é {personalidade}. Valorizas {valores}. Os teus gatilhos emocionais incluem {gatilhos}.
        
        Resumo geral da conversa: {conversation_summary}
        
        {sentiment_info}
        
        A partir desta parte da conversa (lote {batch_number}/{total_batches}), gera um relatório emocional focando-te no Rui.
        Sê específico e inclui:
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
        """Send prompt to the model API and return the response."""
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
        Com base no perfil psicológico do Rui:
        - Personalidade: {personalidade}
        - Valores: {', '.join(valores)}
        - Gatilhos emocionais: {', '.join(gatilhos)}
        - Reflexões recentes: {'; '.join(reflexoes)}
        - Planos recentes: {'; '.join(planos)}
        
        Gere 3 recomendações ESPECÍFICAS e ACIONÁVEIS para melhorar o relacionamento do Rui com a Maria.
        Cada recomendação deve incluir:
        1. O problema ou oportunidade específica
        2. Uma ação concreta para implementar esta semana
        3. Como avaliar se a ação teve efeito positivo
        
        Responda em JSON com um array de objetos, cada um contendo "problema", "acao", e "avaliacao".
        """
        
        try:
            response = self.ask_model(prompt)
            json_text = response.get("choices", [{}])[0].get("text", "{}").strip()
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
            if json_match:
                try:
                    recommendations = json.loads(json_match.group(0))
                    return recommendations.get("recomendacoes", [])
                except json.JSONDecodeError:
                    print("❌ Erro ao parsear JSON das recomendações para Rui")
                    return []
            return []
        except Exception as e:
            print(f"❌ Erro ao gerar recomendações: {str(e)}")
            return []

    def relationship_health_report(self) -> dict:
        """Generate a health report based on relationship metrics over time."""
        metrics = self.memory.get("relationship_metrics", [])
        if not metrics or len(metrics) < 2:
            return {
                "status": "Insuficiente",
                "message": "Dados insuficientes para gerar relatório detalhado",
                "recommendations": []
            }
            
        # Calculate trends
        recent_metrics = metrics[-3:] if len(metrics) >= 3 else metrics
        avg_sentiment = sum(m["sentiment_score"] for m in recent_metrics) / len(recent_metrics)
        avg_balance = sum(m["emotional_balance"] for m in recent_metrics) / len(recent_metrics)
        trend = "estável"
        
        if len(metrics) >= 2:
            if metrics[-1]["emotional_balance"] > metrics[-2]["emotional_balance"]:
                trend = "melhorando"
            elif metrics[-1]["emotional_balance"] < metrics[-2]["emotional_balance"]:
                trend = "piorando"
        
        # Determine health status
        if avg_balance > 2:
            status = "Saudável"
        elif avg_balance > 0:
            status = "Moderadamente saudável"
        elif avg_balance > -2:
            status = "Necessita atenção"
        else:
            status = "Problemático"
            
        # Get recommendations
        recommendations = self.generate_actionable_recommendations()
            
        return {
            "status": status,
            "trend": trend,
            "avg_sentiment": avg_sentiment,
            "avg_balance": avg_balance,
            "message": f"O relacionamento está {status.lower()} e {trend}.",
            "recommendations": recommendations
        }