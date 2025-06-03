"""
APE (Automatic Prompt Engineer) Optimizer
Vollständige Implementierung mit Kandidaten-Evaluation und Selektion

Erweitert die bestehende flow.ipynb Implementierung um:
- Systematische Persona-Evaluation
- Best-Candidate Selection  
- Iterative Verbesserung
- Integration mit bestehender evaluation_metrics.py
- Robustheitsfixes: API-Fehler Validation, Retry-Mechanismus, Detailliertes Logging
"""

import json
import os
import re
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import pandas as pd
from evaluation_metrics import calculate_rouge_scores, calculate_bleu_score, download_nltk_data

# Load environment variables
load_dotenv()

class APEOptimizer:
    """
    Vollständige APE-Implementierung mit Evaluation und Selektion
    
    Erweitert die bestehende Implementierung um:
    - Persona-Kandidaten Evaluation
    - Best-Candidate Selection
    - Systematisches Scoring
    - Robuste API-Fehlerbehandlung mit Retry-Mechanismus
    """
    
    def __init__(self, user_data_dir: Path, results_dir: Path, api_key: Optional[str] = None):
        self.user_data_dir = Path(user_data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Google Gemini API
        if api_key:
            genai.configure(api_key=api_key)
        else:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("API Key nicht in Umgebungsvariablen gefunden. Setzen Sie GOOGLE_API_KEY oder übergeben Sie api_key Parameter.")
            genai.configure(api_key=api_key)
        
        # Initialize evaluation cache and download NLTK data
        self.evaluation_cache = {}
        download_nltk_data()
        
        print("APE Optimizer initialized successfully")
    
    # ===== BESTEHENDE FUNKTIONEN AUS flow.ipynb (übernommen und erweitert) =====
    
    def load_history_tweets_for_user(self, user_id_str: str) -> List[str]:
        """
        Lädt die History-Tweets für einen gegebenen Nutzer aus seiner JSONL-Datei.
        Übernommen aus flow.ipynb
        """
        user_file = self.user_data_dir / f"{user_id_str}.jsonl"
        history_tweet_texts = []

        if not user_file.exists():
            print(f"FEHLER: Datendatei für Nutzer {user_id_str} nicht gefunden unter {user_file}")
            return history_tweet_texts

        try:
            with open(user_file, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f):
                    if line.strip() == "":
                        continue 
                    try:
                        data = json.loads(line)
                        if data.get("set") == "history":
                            tweets = data.get("tweets", [])
                            for tweet in tweets:
                                if 'full_text' in tweet and tweet['full_text']:
                                    history_tweet_texts.append(tweet['full_text'])
                            break 
                    except json.JSONDecodeError as e:
                        print(f"WARNUNG: JSON-Dekodierungsfehler in Zeile {line_number + 1} der Datei {user_file}: {e}")
                        continue
        
            if not history_tweet_texts:
                print(f"WARNUNG: Keine History-Tweets für Nutzer {user_id_str} im 'history'-Set gefunden oder 'full_text' fehlt.")

        except Exception as e:
            print(f"FEHLER beim Lesen der Datei {user_file}: {e}")
        
        return history_tweet_texts
    
    def load_holdout_tweets_for_user(self, user_id_str: str) -> List[Dict]:
        """
        NEU: Lädt die Holdout-Tweets für Evaluation
        Basiert auf load_history_tweets_for_user aber gibt vollständige Tweet-Objekte zurück
        """
        user_file = self.user_data_dir / f"{user_id_str}.jsonl"
        holdout_tweets = []

        if not user_file.exists():
            print(f"FEHLER: Datendatei für Nutzer {user_id_str} nicht gefunden unter {user_file}")
            return holdout_tweets

        try:
            with open(user_file, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f):
                    if line.strip() == "":
                        continue 
                    try:
                        data = json.loads(line)
                        if data.get("set") == "holdout":
                            tweets = data.get("tweets", [])
                            holdout_tweets.extend(tweets)
                            break 
                    except json.JSONDecodeError as e:
                        print(f"WARNUNG: JSON-Dekodierungsfehler in Zeile {line_number + 1} der Datei {user_file}: {e}")
                        continue

        except Exception as e:
            print(f"FEHLER beim Lesen der Datei {user_file}: {e}")
        
        return holdout_tweets

    def format_tweets_for_llm(self, tweet_texts: List[str], max_tweets: int = 50, max_chars: int = 10000) -> str:
        """
        Formatiert eine Liste von Tweet-Texten zu einem einzigen String für den LLM-Input.
        Übernommen aus flow.ipynb
        """
        selected_tweets = tweet_texts[:max_tweets]
        
        formatted_string = ""
        for tweet in selected_tweets:
            if len(formatted_string) + len(tweet) + len("\n---\n") > max_chars:
                break
            formatted_string += tweet + "\n---\n"
            
        if not formatted_string and tweet_texts: 
            print(f"WARNUNG: Konnte Tweets nicht formatieren. Möglicherweise sind einzelne Tweets zu lang oder max_chars zu klein.")
            if tweet_texts[0] and len(tweet_texts[0]) <= max_chars:
                return tweet_texts[0]
            return ""

        return formatted_string.strip().rstrip("-")

    def get_enhanced_meta_prompt_ape(self, num_candidates: int) -> str:
        """
        VERBESSERT: Deutlich verbesserter Meta-Prompt für APE mit Beispielen und Qualitätskriterien
        Ersetzt get_meta_prompt_for_persona_generation_json aus flow.ipynb
        """
        return f"""
Sie sind ein Experte für Social-Media-Analyse und erstellen Persona-Beschreibungen für präzise Stil-Imitation.

AUFGABE: Analysieren Sie die folgenden Tweets und erstellen Sie {num_candidates} verschiedene, 
hochspezifische Persona-Beschreibungen, die ein KI-System befähigen würden, den Schreibstil 
dieses Nutzers mit maximaler Genauigkeit zu imitieren.

QUALITÄTSKRITERIEN für exzellente Personas:

1. SPEZIFIZITÄT: Verwenden Sie konkrete, messbare Beschreibungen
   ✅ Gut: "Nutzt 2-3 Emojis pro Tweet, bevorzugt 🤔💭🔥"
   ❌ Schlecht: "Nutzt manchmal Emojis"

2. STILMERKMALE: Fokus auf erkennbare sprachliche Muster
   ✅ Gut: "Beginnt Fragen oft mit 'Mal ehrlich,' oder 'Verstehe ich das richtig,'"
   ❌ Schlecht: "Stellt gelegentlich Fragen"

3. STRUKTURELLE MUSTER: Satzlänge, Interpunktion, Formatierung
   ✅ Gut: "Schreibt meist in 2-3 kurzen Sätzen (8-12 Wörter), trennt Gedanken mit '...' ab"
   ❌ Schlecht: "Schreibt relativ kurz"

4. THEMATISCHE PRÄFERENZEN: Konkrete Interessensgebiete und Perspektiven
   ✅ Gut: "Diskutiert hauptsächlich Tech-Startups und Krypto, oft kritisch gegenüber Hype-Trends"
   ❌ Schlecht: "Interessiert sich für Technologie"

5. INTERAKTIONSSTIL: Wie der Nutzer auf andere reagiert
   ✅ Gut: "Bei Meinungsverschiedenheiten verwendet 'Ich sehe das anders weil...' statt direkter Widerspruch"
   ❌ Schlecht: "Höflich in Diskussionen"

BEISPIEL einer optimalen Persona-Beschreibung:
"Dieser Nutzer schreibt prägnant in 1-2 Sätzen pro Tweet (durchschnittlich 15-25 Wörter), 
verwendet häufig rhetorische Fragen am Ende ('...oder sehe ich das falsch?'), bevorzugt 
konkrete Zahlen und Fakten in Argumentationen, hat einen leicht ironischen aber nicht 
sarkastischen Ton, diskutiert primär über Klimapolitik und erneuerbare Energien mit 
technik-optimistischer Grundhaltung, und leitet neue Gedanken oft mit 'Übrigens,' oder 
'Nebenbei bemerkt,' ein."

AUSGABEFORMAT:
Ihre Antwort MUSS ein valides JSON-Objekt sein:
{{
  "persona_descriptions": [
    "Erste detaillierte Persona-Beschreibung hier...",
    "Zweite detaillierte Persona-Beschreibung hier...",
    ...
  ]
}}

Analysieren Sie jetzt die folgenden Tweets und erstellen Sie {num_candidates} verschiedene, 
hochspezifische Persona-Beschreibungen:

"""

    def call_gemini_api_for_persona_gen(self, full_prompt_for_llm: str, num_candidates: int) -> List[str]:
        """
        Ruft die Google Gemini API auf, um Persona-Prompt-Kandidaten als JSON zu generieren.
        Übernommen aus flow.ipynb mit leichten Verbesserungen
        """
        print("\n--- SENDE PROMPT AN GOOGLE GEMINI API (JSON Modus) ---")
        print("--- WARTE AUF ANTWORT VON GEMINI API ... ---")

        try:
            model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')

            generation_config = genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=2048,
                temperature=0.7,
                response_mime_type="application/json"
            )

            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            response = model.generate_content(
                contents=full_prompt_for_llm,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            print("--- ANTWORT VON GEMINI API ERHALTEN (JSON Modus) ---")

            if not response.text:
                print("WARNUNG: Kein Text von der Gemini API erhalten (JSON Modus).")
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    print(f"Blockierungsgrund: {response.prompt_feedback.block_reason}")
                return []
        
            try:
                parsed_json = json.loads(response.text)
                raw_descriptions = parsed_json.get("persona_descriptions")

                if not isinstance(raw_descriptions, list):
                    print(f"WARNUNG: JSON-Struktur unerwartet. 'persona_descriptions' ist keine Liste oder fehlt. Erhalten: {parsed_json}")
                    if isinstance(parsed_json, str):
                         return [f"Persona 1: {parsed_json.strip()}"]
                    return []

                persona_candidates_from_json = [str(desc).strip() for desc in raw_descriptions if isinstance(desc, str) and str(desc).strip()]
                
                if not persona_candidates_from_json and raw_descriptions:
                     print(f"WARNUNG: 'persona_descriptions' enthielt Elemente, aber keine validen Strings nach Filterung. Original: {raw_descriptions}")
                     return []
                
                if not persona_candidates_from_json:
                    print("WARNUNG: Keine Persona-Beschreibungen im JSON gefunden.")
                    return []

                # Formatieren der Beschreibungen zu "Persona X: ..." für Konsistenz
                formatted_candidates = []
                for i, desc in enumerate(persona_candidates_from_json[:num_candidates]):
                    cleaned_desc = re.sub(r"^(Persona\s*\d*\s*:)\s*", "", desc, flags=re.IGNORECASE).strip()
                    if cleaned_desc:
                        formatted_candidates.append(f"Persona {i + 1}: {cleaned_desc}")
                
                if not formatted_candidates and persona_candidates_from_json:
                     print("WARNUNG: Nach der Formatierung waren keine Kandidaten mehr übrig. Verwende Roh-JSON-Strings.")
                     return [f"Persona {i+1}: {desc_orig}" for i, desc_orig in enumerate(persona_candidates_from_json[:num_candidates])]

                print(f"--- {len(formatted_candidates)} Persona-Kandidaten erfolgreich aus JSON geparst und formatiert ---")
                return formatted_candidates

            except json.JSONDecodeError as e:
                print(f"FEHLER: Konnte JSON-Antwort von Gemini nicht parsen: {e}")
                print(f"Rohtext, der zum Fehler führte:\n{response.text}")
                return []

        except Exception as e:
            print(f"FEHLER bei der Kommunikation mit der Gemini API (JSON Modus): {e}")
            return []

    # ===== NEUE APE-OPTIMIERUNG FUNKTIONEN =====

    def generate_persona_candidates(self, user_id_str: str, num_candidates: int = 5) -> List[str]:
        """
        Generiert Persona-Kandidaten (übernommen aus flow.ipynb aber als separate Methode)
        """
        print(f"Starte Generierung von Persona-Prompt-Kandidaten für Nutzer: {user_id_str}")
        
        history_tweets = self.load_history_tweets_for_user(user_id_str)
        if not history_tweets:
            print(f"Konnte keine History-Tweets für {user_id_str} laden. Überspringe Persona-Generierung.")
            return []

        formatted_history_for_llm = self.format_tweets_for_llm(history_tweets)
        if not formatted_history_for_llm:
            print(f"Konnte History-Tweets für {user_id_str} nicht für LLM formatieren. Überspringe Persona-Generierung.")
            return []

        meta_prompt = self.get_enhanced_meta_prompt_ape(num_candidates)
        full_prompt_for_llm = meta_prompt + "\n\n" + formatted_history_for_llm

        persona_candidates = self.call_gemini_api_for_persona_gen(full_prompt_for_llm, num_candidates)
        
        print(f"Generierung für Nutzer {user_id_str} abgeschlossen. {len(persona_candidates)} Kandidaten erhalten.")
        return persona_candidates

    # ===== ROBUSTHEITSFIXES: API-FEHLER VALIDATION & RETRY =====

    def ist_api_call_fehlgeschlagen(self, response: any) -> tuple[bool, str | None, dict | None]:
        """
        Prüft, ob ein API-Aufruf fehlgeschlagen ist, basierend auf dem HTTP-Statuscode 
        und dem Vorhandensein eines Fehlerobjekts im JSON-Body.
        
        Args:
            response: Das Antwortobjekt des API-Aufrufs (Gemini API Response Object).
            
        Returns:
            Ein Tupel bestehend aus:
            - bool: True, wenn ein Fehler aufgetreten ist, sonst False.
            - str | None: Eine Fehlermeldung, falls ein Fehler erkannt wurde, sonst None.
            - dict | None: Das detaillierte Fehlerobjekt, falls vorhanden, sonst None.
        """
        # Für Gemini API: Prüfe auf empty response oder prompt feedback errors
        if not hasattr(response, 'text') or not response.text:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                    fehlermeldung = f"API-Fehler: Content blocked - {response.prompt_feedback.block_reason}"
                    return True, fehlermeldung, {"block_reason": str(response.prompt_feedback.block_reason)}
            
            return True, "API-Fehler: Leere Antwort erhalten", None
        
        # Prüfe ob die Antwort ein JSON-Error ist (manchmal gibt Gemini JSON-Errors zurück)
        response_text = response.text.strip()
        if response_text.startswith('{'):
            try:
                json_response = json.loads(response_text)
                if isinstance(json_response, dict) and "error" in json_response:
                    error_details = json_response["error"]
                    
                    # Präzisere Fehlermeldung aus dem JSON-Error-Objekt
                    if isinstance(error_details, dict) and "message" in error_details:
                        fehlermeldung = f"API-Fehler: {error_details.get('message')}"
                        if "code" in error_details:
                            fehlermeldung += f" (Code: {error_details.get('code')})"
                    elif isinstance(error_details, str):
                        fehlermeldung = f"API-Fehler: {error_details}"
                    else:
                        fehlermeldung = "API-Fehler: Unbekannter Fehler in JSON-Response"
                    
                    return True, fehlermeldung, error_details
                    
                # Prüfe auf direkten Error (ohne "error" wrapper)
                elif isinstance(json_response, dict) and "message" in json_response and "code" in json_response:
                    fehlermeldung = f"API-Fehler: {json_response.get('message')} (Code: {json_response.get('code')})"
                    return True, fehlermeldung, json_response
                    
            except json.JSONDecodeError:
                # Wenn es mit { anfängt aber kein valides JSON ist, könnte es trotzdem ein Error sein
                pass
        
        # Kein Fehler erkannt
        return False, None, None

    def _call_api_with_logging(self, prompt: str, max_tokens: int = 280) -> tuple[any, str]:
        """
        NEU: API-Call mit detailliertem Logging und Fehlerbehandlung
        Returns: (response_object, result_text)
        """
        print(f"        Making API call (max_tokens: {max_tokens})...")
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7
            )
            
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            response = model.generate_content(
                contents=prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Prüfe auf API-Fehler in der Response
            if not response.text:
                print(f"        API returned empty response")
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    print(f"        Block reason: {response.prompt_feedback.block_reason}")
                return response, ""
            
            result = response.text.strip()
            print(f"        API call successful, received {len(result)} characters")
            
            # Basis-Bereinigung (nur technische Artefakte, nicht Inhalt)
            result = result.replace('"', '').replace("'", "'")
            
            return response, result
            
        except Exception as e:
            # Hier werden echte API-Exceptions gefangen (Rate Limits, Network Errors, etc.)
            print(f"        API call failed with exception: {type(e).__name__}: {e}")
            return None, ""

    def generate_reply_imitation_with_validation(self, persona: str, holdout_tweet: Dict) -> tuple[any, str]:
        """
        VERBESSERT: Reply-Generierung mit API-Fehler Validation
        Returns: (response_object, result_text)
        """
        previous_msg = holdout_tweet.get('previous_message', '')
        
        prompt = f"""
{persona}

Previous message in conversation:
{previous_msg}

Generate a response that this persona would write. The response should be natural, 
authentic to the persona's style, and appropriately respond to the previous message.
Write only the response, nothing else.

Response:"""
        
        print(f"        Generating reply imitation...")
        response, result = self._call_api_with_logging(prompt, max_tokens=280)
        
        # Zusätzliche Bereinigung
        if result:
            result = result.replace("Response:", "").strip()
        
        return response, result

    def generate_completion_imitation_with_validation(self, persona: str, holdout_tweet: Dict) -> tuple[any, str]:
        """
        VERBESSERT: Completion-Generierung mit API-Fehler Validation
        Returns: (response_object, result_text)
        """
        original_text = holdout_tweet.get('full_text', '')
        if not original_text:
            print(f"        No original text found for completion")
            return None, ""
            
        words = original_text.split()
        
        # Nutze erste 30% der Wörter als Fragment (mindestens 3, maximal 10)
        fragment_length = max(3, min(10, len(words) // 3))
        if fragment_length >= len(words):
            print(f"        SKIPPED: Tweet too short for completion ({len(words)} words). Minimum fragment length is {fragment_length}.")
            return None, ""
            
        fragment = ' '.join(words[:fragment_length])
        
        prompt = f"""
{persona}

Complete this tweet fragment in the exact style of this persona:

Fragment: "{fragment}"

Complete the tweet (make it natural and authentic to the persona):
{fragment}"""
        
        print(f"        Generating completion for fragment: '{fragment}'...")
        response, result = self._call_api_with_logging(prompt, max_tokens=280)
        
        # Stelle sicher, dass das Fragment enthalten ist
        if result and not result.lower().startswith(fragment.lower()):
            result = fragment + " " + result
        
        return response, result

    def generate_test_imitation_with_retry(self, persona: str, holdout_tweet: Dict, max_retries: int = 3) -> str:
        """
        VERBESSERT: Test-Imitation mit Retry-Mechanismus und API-Fehler Validation
        """
        for attempt in range(max_retries):
            try:
                print(f"      Attempt {attempt + 1}/{max_retries}...")
                
                # Prüfe ob es sich um eine Reply handelt
                # Determine imitation type and log relevant input
                if holdout_tweet.get('reply_to_id') and holdout_tweet.get('previous_message'):
                    print(f"      Attempting REPLY imitation.")
                    print(f"        Previous message for reply: '{holdout_tweet.get('previous_message', '')}'")
                    response, result = self.generate_reply_imitation_with_validation(persona, holdout_tweet)
                else:
                    print(f"      Attempting COMPLETION imitation.")
                    print(f"        Original text for completion: '{holdout_tweet.get('full_text', '')}'")
                    response, result = self.generate_completion_imitation_with_validation(persona, holdout_tweet)
                
                # Log raw response and processed result
                print(f"      Raw API response text: '{response.text if response else 'None'}'")
                print(f"      Processed result before length check: '{result}' (length: {len(result.strip())})")
                
                # Validiere ob API-Call fehlgeschlagen ist
                ist_fehler, fehlermeldung, error_details = self.ist_api_call_fehlgeschlagen(response)
                
                if ist_fehler:
                    print(f"      Attempt {attempt + 1}: API-Fehler erkannt: {fehlermeldung}")
                    if error_details:
                        print(f"        Error details: {error_details}")
                    if attempt < max_retries - 1:
                        print(f"      Retrying...")
                        continue
                    else:
                        print(f"      Max retries reached, API-Fehler persistent")
                        return ""
                
                # Erfolgreiche Generierung
                if result and len(result.strip()) >= 10:
                    print(f"      Attempt {attempt + 1}: Success! Generated {len(result)} characters")
                    return result.strip()
                else:
                    print(f"      Attempt {attempt + 1}: Generated text too short or empty (length: {len(result.strip())})")
                    if attempt < max_retries - 1:
                        print(f"      Retrying...")
                        continue
                        
            except Exception as e:
                print(f"      Attempt {attempt + 1}: Exception occurred: {e}")
                if attempt < max_retries - 1:
                    print(f"      Retrying...")
                    continue
                else:
                    print(f"      Max retries reached after exception")
                    return ""
        
        print(f"      All {max_retries} attempts failed")
        return ""

    def evaluate_persona_candidate_with_logging(self, persona: str, holdout_tweets: List[Dict], user_id: str, num_test_tweets: int = 10) -> float:
        """
        VERBESSERT: Persona-Evaluation mit detailliertem Logging
        """
        # Cache-Key für wiederverwendbare Evaluierungen
        cache_key = f"{user_id}_{hash(persona)}_{num_test_tweets}"
        if cache_key in self.evaluation_cache:
            print(f"  Using cached result for persona evaluation")
            return self.evaluation_cache[cache_key]
        
        test_tweets = holdout_tweets[:num_test_tweets]
        total_rouge1 = 0
        total_rouge2 = 0
        total_rougeL = 0
        total_bleu = 0
        successful_imitations = 0
        
        print(f"  Evaluating persona with {len(test_tweets)} test tweets...")
        print(f"  Persona preview: {persona[:100]}...")
        
        for i, holdout_tweet in enumerate(test_tweets):
            print(f"    Processing tweet {i+1}/{len(test_tweets)}:")
            original_full_text = holdout_tweet.get('full_text', '')
            previous_message = holdout_tweet.get('previous_message', '')
            print(f"      Original Tweet (full_text): '{original_full_text}'")
            if previous_message:
                print(f"      Previous Message: '{previous_message}'")
            
            try:
                # Generiere Test-Imitation mit Retry-Mechanismus
                imitation = self.generate_test_imitation_with_retry(persona, holdout_tweet)
                
                if not imitation or len(imitation.strip()) < 10:
                    print(f"      Result: FAILED - No valid imitation generated")
                    continue
                
                # Bewerte Imitation gegen Original
                original_text = holdout_tweet.get('full_text', '')
                if not original_text:
                    print(f"      Result: SKIPPED - No original text")
                    continue
                
                rouge_scores = calculate_rouge_scores(original_text, imitation)
                bleu_score = calculate_bleu_score(original_text, imitation)
                
                total_rouge1 += rouge_scores['rouge1_fscore']
                total_rouge2 += rouge_scores['rouge2_fscore']
                total_rougeL += rouge_scores['rougeL_fscore']
                total_bleu += bleu_score
                successful_imitations += 1
                
                print(f"      Generated: {imitation[:50]}...")
                print(f"      Scores: R1={rouge_scores['rouge1_fscore']:.3f}, R2={rouge_scores['rouge2_fscore']:.3f}, RL={rouge_scores['rougeL_fscore']:.3f}, BLEU={bleu_score:.3f}")
                print(f"      Result: SUCCESS ✓")
                
            except Exception as e:
                print(f"      Result: ERROR - {e}")
                continue
        
        if successful_imitations == 0:
            print(f"  EVALUATION FAILED: No successful imitations generated!")
            return 0.0
        
        # Berechne Durchschnittswerte
        avg_rouge1 = total_rouge1 / successful_imitations
        avg_rouge2 = total_rouge2 / successful_imitations
        avg_rougeL = total_rougeL / successful_imitations
        avg_bleu = total_bleu / successful_imitations
        
        # Calculate combined score (average of ROUGE-1, ROUGE-2, ROUGE-L, and BLEU)
        combined_score = (avg_rouge1 + avg_rouge2 + avg_rougeL + avg_bleu) / 4
        
        success_rate = successful_imitations / len(test_tweets)
        
        print(f"  EVALUATION SUMMARY:")
        print(f"    Successful imitations: {successful_imitations}/{len(test_tweets)} ({success_rate:.1%})")
        print(f"    Average ROUGE-1: {avg_rouge1:.4f}")
        print(f"    Average ROUGE-2: {avg_rouge2:.4f}")
        print(f"    Average ROUGE-L: {avg_rougeL:.4f}")
        print(f"    Average BLEU: {avg_bleu:.4f}")
        print(f"    Combined score: {combined_score:.4f}")
        
        # Cache das Ergebnis
        self.evaluation_cache[cache_key] = combined_score
        
        return combined_score

    # ===== LEGACY SUPPORT (für alte Funktionsaufrufe) =====
    
    def generate_test_imitation(self, persona: str, holdout_tweet: Dict) -> str:
        """Legacy-Support: Leitet an die neue Retry-Funktion weiter"""
        return self.generate_test_imitation_with_retry(persona, holdout_tweet)
    
    def evaluate_persona_candidate(self, persona: str, holdout_tweets: List[Dict], user_id: str, num_test_tweets: int = 10) -> float:
        """Legacy-Support: Leitet an die neue Logging-Funktion weiter"""
        return self.evaluate_persona_candidate_with_logging(persona, holdout_tweets, user_id, num_test_tweets)

    # ===== HAUPTFUNKTIONEN =====

    def run_ape_optimization(self, user_id: str, num_candidates: int = 5, num_evaluation_tweets: int = 10) -> Tuple[Optional[str], float, Dict]:
        """
        NEU: Vollständiger APE-Prozess mit Evaluation und Selektion
        
        Args:
            user_id: ID des zu verarbeitenden Nutzers
            num_candidates: Anzahl der zu generierenden Persona-Kandidaten
            num_evaluation_tweets: Anzahl der Holdout-Tweets für Evaluation
            
        Returns:
            Tuple von (best_persona, best_score, detailed_results)
        """
        print(f"\n{'='*60}")
        print(f"STARTING APE OPTIMIZATION FOR USER: {user_id}")
        print(f"{'='*60}")
        
        # 1. Lade Nutzer-Daten
        history_tweets = self.load_history_tweets_for_user(user_id)
        holdout_tweets = self.load_holdout_tweets_for_user(user_id)
        
        if not history_tweets:
            print(f"ERROR: No history tweets found for user {user_id}")
            return None, 0.0, {}
            
        if not holdout_tweets:
            print(f"ERROR: No holdout tweets found for user {user_id}")
            return None, 0.0, {}
        
        print(f"Loaded {len(history_tweets)} history tweets and {len(holdout_tweets)} holdout tweets")
        
        # 2. Generiere Persona-Kandidaten
        print(f"\nStep 1: Generating {num_candidates} persona candidates...")
        candidates = self.generate_persona_candidates(user_id, num_candidates)
        
        if not candidates:
            print(f"ERROR: Failed to generate persona candidates for user {user_id}")
            return None, 0.0, {}
        
        print(f"Generated {len(candidates)} candidates successfully")
        
        # 3. Evaluiere jeden Kandidaten
        print(f"\nStep 2: Evaluating persona candidates...")
        candidate_results = []
        
        for i, candidate in enumerate(candidates):
            print(f"\nEvaluating candidate {i+1}/{len(candidates)}:")
            print(f"Persona: {candidate[:100]}...")
            
            score = self.evaluate_persona_candidate(
                candidate, holdout_tweets, user_id, num_evaluation_tweets
            )
            
            candidate_results.append({
                'persona': candidate,
                'score': score,
                'rank': i + 1,
                'user_id': user_id
            })
            
            print(f"Candidate {i+1} final score: {score:.4f}")
        
        # 4. Wähle besten Kandidaten
        print(f"\nStep 3: Selecting best candidate...")
        best_result = max(candidate_results, key=lambda x: x['score'])
        best_persona = best_result['persona']
        best_score = best_result['score']
        
        # Sortiere Ergebnisse nach Score
        candidate_results.sort(key=lambda x: x['score'], reverse=True)
        for i, result in enumerate(candidate_results):
            result['final_rank'] = i + 1
        
        print(f"\nBEST PERSONA SELECTED!")
        print(f"Score: {best_score:.4f}")
        print(f"Persona: {best_persona}")
        
        # 5. Speichere detaillierte Ergebnisse
        detailed_results = {
            'user_id': user_id,
            'timestamp': str(datetime.now()),
            'best_persona': best_persona,
            'best_score': best_score,
            'num_candidates_generated': len(candidates),
            'num_evaluation_tweets': num_evaluation_tweets,
            'all_candidates': candidate_results,
            'statistics': {
                'mean_score': sum(r['score'] for r in candidate_results) / len(candidate_results),
                'std_score': (sum((r['score'] - sum(r2['score'] for r2 in candidate_results) / len(candidate_results))**2 for r in candidate_results) / len(candidate_results))**0.5,
                'min_score': min(r['score'] for r in candidate_results),
                'max_score': max(r['score'] for r in candidate_results)
            }
        }
        
        # Speichere Ergebnisse
        self.save_ape_results(user_id, detailed_results)
        
        return best_persona, best_score, detailed_results

    def save_ape_results(self, user_id: str, results: Dict):
        """
        NEU: Speichert APE-Optimierungsergebnisse
        """
        # Einzelne Nutzer-Ergebnisse
        user_results_file = self.results_dir / f"ape_results_{user_id}.json"
        try:
            with open(user_results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {user_results_file}")
        except Exception as e:
            print(f"Error saving user results: {e}")
        
        # Sammle alle Ergebnisse in einer Master-Datei
        master_results_file = self.results_dir / "ape_optimization_results.json"
        all_results = {}
        
        # Lade bestehende Ergebnisse falls vorhanden
        if master_results_file.exists():
            try:
                with open(master_results_file, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
            except:
                pass
        
        # Füge neue Ergebnisse hinzu
        all_results[user_id] = results
        
        try:
            with open(master_results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"Master results updated: {master_results_file}")
        except Exception as e:
            print(f"Error updating master results: {e}")

    def run_batch_optimization(self, user_ids: List[str], num_candidates: int = 5, num_evaluation_tweets: int = 10) -> Dict:
        """
        NEU: Führt APE-Optimierung für mehrere Nutzer aus
        """
        print(f"\n{'='*80}")
        print(f"STARTING BATCH APE OPTIMIZATION FOR {len(user_ids)} USERS")
        print(f"{'='*80}")
        
        batch_results = {}
        successful_optimizations = 0
        
        for i, user_id in enumerate(user_ids):
            print(f"\n[{i+1}/{len(user_ids)}] Processing user: {user_id}")
            
            try:
                best_persona, best_score, detailed_results = self.run_ape_optimization(
                    user_id, num_candidates, num_evaluation_tweets
                )
                
                if best_persona:
                    batch_results[user_id] = {
                        'success': True,
                        'best_persona': best_persona,
                        'best_score': best_score,
                        'detailed_results': detailed_results
                    }
                    successful_optimizations += 1
                    print(f"✅ User {user_id} completed successfully (Score: {best_score:.4f})")
                else:
                    batch_results[user_id] = {
                        'success': False,
                        'error': 'Failed to generate or evaluate personas'
                    }
                    print(f"❌ User {user_id} failed")
                    
            except Exception as e:
                batch_results[user_id] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"❌ User {user_id} failed with error: {e}")
        
        print(f"\n{'='*80}")
        print(f"BATCH OPTIMIZATION COMPLETED")
        print(f"Successful: {successful_optimizations}/{len(user_ids)} users")
        print(f"Success rate: {successful_optimizations/len(user_ids)*100:.1f}%")
        
        # Speichere Batch-Ergebnisse
        batch_summary = {
            'timestamp': str(datetime.now()),
            'total_users': len(user_ids),
            'successful_users': successful_optimizations,
            'success_rate': successful_optimizations/len(user_ids),
            'parameters': {
                'num_candidates': num_candidates,
                'num_evaluation_tweets': num_evaluation_tweets
            },
            'results': batch_results
        }
        
        batch_file = self.results_dir / f"ape_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch_summary, f, indent=2, ensure_ascii=False)
            print(f"Batch results saved to: {batch_file}")
        except Exception as e:
            print(f"Error saving batch results: {e}")
        
        return batch_summary


def main():
    """
    Hauptfunktion - Beispiel für die Nutzung des APE Optimizers
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='APE (Automatic Prompt Engineer) Optimizer')
    parser.add_argument('--user_data_dir', type=str, required=True,
                       help='Verzeichnis mit Nutzer-JSONL-Dateien')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Ausgabeverzeichnis für Ergebnisse')
    parser.add_argument('--user_id', type=str,
                       help='Spezifische User-ID für Einzelverarbeitung')
    parser.add_argument('--eligible_users_file', type=str,
                       help='JSON-Datei mit Liste von Nutzer-IDs für Batch-Verarbeitung')
    parser.add_argument('--num_candidates', type=int, default=5,
                       help='Anzahl der zu generierenden Persona-Kandidaten (default: 5)')
    parser.add_argument('--num_evaluation_tweets', type=int, default=10,
                       help='Anzahl der Holdout-Tweets für Evaluation (default: 10)')
    parser.add_argument('--api_key', type=str,
                       help='Google Gemini API Key (optional, kann auch über Umgebungsvariable gesetzt werden)')
    
    args = parser.parse_args()
    
    try:
        # Initialisiere APE Optimizer
        optimizer = APEOptimizer(
            user_data_dir=args.user_data_dir,
            results_dir=args.results_dir,
            api_key=args.api_key
        )
        
        if args.user_id:
            # Einzelnutzer-Modus
            print(f"Running APE optimization for single user: {args.user_id}")
            best_persona, best_score, results = optimizer.run_ape_optimization(
                user_id=args.user_id,
                num_candidates=args.num_candidates,
                num_evaluation_tweets=args.num_evaluation_tweets
            )
            
            if best_persona:
                print(f"\n🎉 OPTIMIZATION SUCCESSFUL!")
                print(f"Best Score: {best_score:.4f}")
                print(f"Best Persona: {best_persona}")
            else:
                print(f"\n❌ OPTIMIZATION FAILED for user {args.user_id}")
                
        elif args.eligible_users_file:
            # Batch-Modus
            try:
                with open(args.eligible_users_file, 'r', encoding='utf-8') as f:
                    user_ids = json.load(f)
                
                if not isinstance(user_ids, list):
                    print(f"ERROR: {args.eligible_users_file} should contain a list of user IDs")
                    return
                
                print(f"Running APE optimization for {len(user_ids)} users from {args.eligible_users_file}")
                batch_results = optimizer.run_batch_optimization(
                    user_ids=user_ids,
                    num_candidates=args.num_candidates,
                    num_evaluation_tweets=args.num_evaluation_tweets
                )
                
                print(f"\n🎉 BATCH OPTIMIZATION COMPLETED!")
                print(f"Success rate: {batch_results['success_rate']*100:.1f}%")
                
            except FileNotFoundError:
                print(f"ERROR: File {args.eligible_users_file} not found")
            except json.JSONDecodeError:
                print(f"ERROR: Invalid JSON in {args.eligible_users_file}")
        else:
            print("ERROR: Either --user_id or --eligible_users_file must be provided")
            parser.print_help()
            
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        raise


# ===== ZUSÄTZLICHE UTILITY-FUNKTIONEN =====

def analyze_ape_results(results_dir: str):
    """
    Utility-Funktion zur Analyse der APE-Ergebnisse
    """
    results_path = Path(results_dir)
    master_file = results_path / "ape_optimization_results.json"
    
    if not master_file.exists():
        print(f"No results found at {master_file}")
        return
    
    try:
        with open(master_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    print(f"\n{'='*60}")
    print(f"APE OPTIMIZATION RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    scores = []
    user_count = len(all_results)
    
    print(f"Total users optimized: {user_count}")
    
    for user_id, result in all_results.items():
        score = result.get('best_score', 0)
        scores.append(score)
        print(f"User {user_id}: {score:.4f}")
    
    if scores:
        import statistics
        print(f"\nScore Statistics:")
        print(f"Mean: {statistics.mean(scores):.4f}")
        print(f"Median: {statistics.median(scores):.4f}")
        print(f"Min: {min(scores):.4f}")
        print(f"Max: {max(scores):.4f}")
        print(f"Std Dev: {statistics.stdev(scores):.4f}")
        
        # Top 5 performers
        sorted_results = sorted(all_results.items(), 
                              key=lambda x: x[1].get('best_score', 0), 
                              reverse=True)
        
        print(f"\nTop 5 Performers:")
        for i, (user_id, result) in enumerate(sorted_results[:5]):
            score = result.get('best_score', 0)
            persona = result.get('best_persona', '')[:100] + "..."
            print(f"{i+1}. User {user_id}: {score:.4f}")
            print(f"   Persona: {persona}")


def compare_personas_quality(results_dir: str, user_id: str):
    """
    Utility-Funktion zum detaillierten Vergleich der Persona-Kandidaten eines Nutzers
    """
    results_path = Path(results_dir)
    user_file = results_path / f"ape_results_{user_id}.json"
    
    if not user_file.exists():
        print(f"No results found for user {user_id} at {user_file}")
        return
    
    try:
        with open(user_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    print(f"\n{'='*80}")
    print(f"PERSONA QUALITY COMPARISON FOR USER: {user_id}")
    print(f"{'='*80}")
    
    candidates = results.get('all_candidates', [])
    if not candidates:
        print("No candidate data found")
        return
    
    for i, candidate in enumerate(candidates):
        rank = candidate.get('final_rank', i+1)
        score = candidate.get('score', 0)
        persona = candidate.get('persona', '')
        
        print(f"\nRank {rank} (Score: {score:.4f}):")
        print(f"Persona: {persona}")
        print("-" * 80)


if __name__ == "__main__":
    main()