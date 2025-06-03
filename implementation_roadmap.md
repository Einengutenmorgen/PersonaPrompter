# Implementierungs-Roadmap: Social Media User Imitation

## Aktueller Status und Ziele

### ✅ Implementiert (APE-Basis)
- Datenaufbereitung und Nutzerfilterung
- Google Gemini API Integration
- JSON-strukturierte Persona-Generierung 
- Basis Reply- und Completion-Strategien
- ROUGE/BLEU Evaluierung

### 🎯 Zu Implementieren
1. **OPRO (Optimization by PROmpting)**
2. **EvoPrompt (Evolutionary Prompt Optimization)**
3. **Verbesserung der Completion-Strategie**
4. **Tweet/Link-Filterung mit Thread-Vollständigkeit**
5. **APE-Implementierung Review & Optimierung**

---

## Phase 1: APE Review & Optimierung (Woche 1-2)

### 1.1 Code-Review der aktuellen APE-Implementierung

#### Zu überprüfende Komponenten:
```python
# flow.ipynb - Kritische Funktionen:
- generate_persona_candidates_for_user()
- call_gemini_api_for_persona_gen()
- get_meta_prompt_for_persona_generation_json()
- format_tweets_for_llm()
```

#### Review-Kriterien:
- **Prompt-Qualität**: Ist der Meta-Prompt optimal für Persona-Extraktion?
- **API-Effizienz**: Werden API-Calls optimal genutzt?
- **Kandidaten-Diversität**: Erzeugt APE ausreichend diverse Personas?
- **Fehlerbehandlung**: Robustheit bei API-Fehlern und Edge Cases
- **Evaluierung**: Wird die beste Persona korrekt ausgewählt?

#### Identifizierte Verbesserungen:
1. **Fehlende Persona-Selektion**: APE generiert Kandidaten, aber wählt nicht den besten aus
2. **Begrenzte Kandidaten-Evaluation**: Keine systematische Bewertung der Personas
3. **Statische Prompt-Struktur**: Kein Learning aus erfolgreichen Prompts

### 1.2 APE-Optimierungen

#### A) Kandidaten-Evaluierung implementieren:
```python
def evaluate_persona_candidates(candidates, holdout_tweets, user_id):
    """
    Bewertet jeden Persona-Kandidaten durch:
    1. Generierung von Test-Imitationen
    2. ROUGE/BLEU-Scoring gegen Holdout-Set
    3. Auswahl des besten Kandidaten
    """
    best_candidate = None
    best_score = 0
    
    for candidate in candidates:
        test_imitations = generate_test_imitations(candidate, holdout_tweets)
        avg_score = evaluate_imitations(test_imitations, holdout_tweets)
        
        if avg_score > best_score:
            best_score = avg_score
            best_candidate = candidate
    
    return best_candidate, best_score
```

#### B) Meta-Prompt Verfeinerung:
```python
def get_optimized_meta_prompt_ape(user_analysis_context):
    """
    Verbesserte Prompt-Struktur mit:
    - Klareren Anweisungen für Persona-Aspekte
    - Beispielen für gute Persona-Beschreibungen
    - Spezifischen Evaluierungskriterien
    """
```

---

## Phase 2: Link-Filterung & Completion-Verbesserung (Woche 2-3)

### 2.1 Intelligente Tweet-Filterung

#### Anforderungen:
- **Tweet/Reply-Auswahl**: Filtere Tweets mit Links für History/Holdout
- **Thread-Vollständigkeit**: Alle Nachrichten in Threads behalten
- **URL-Erkennung**: Robuste Link-Detection (bit.ly, t.co, http/https)

#### Implementierung:
```python
def filter_tweets_for_personas(tweets_df, preserve_threads=True):
    """
    Filtert Tweets mit Links für Persona-Training,
    erhält aber Thread-Vollständigkeit.
    """
    # Link-Detection Regex
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Filtere für History/Holdout
    clean_tweets = tweets_df[~tweets_df['full_text'].str.contains(url_pattern, na=False)]
    
    # Threads separat behandeln (alle Nachrichten behalten)
    if preserve_threads:
        thread_tweets = extract_complete_threads(tweets_df)
        return clean_tweets, thread_tweets
    
    return clean_tweets
```

### 2.2 Erweiterte Completion-Strategie

#### Aktuelle Limitation:
- Nur erste 7 Wörter als Fragment
- Keine semantische Analyse der Fragmente
- Keine Berücksichtigung von Tweet-Struktur

#### Neue Completion-Ansätze:

#### A) Semantisch-basierte Fragmente:
```python
def generate_semantic_fragments(tweet_text, fragment_strategies):
    """
    Verschiedene Fragment-Strategien:
    1. Thema-basiert: Bis zum ersten semantischen Break
    2. Emotion-basiert: Bis zum Stimmungswechsel  
    3. Struktur-basiert: Bis zu Hashtags/Mentions
    4. Längen-variabel: 3-15 Wörter je nach Kontext
    """
    fragments = []
    
    # Strategie 1: Semantische Segmentierung
    semantic_breaks = identify_semantic_boundaries(tweet_text)
    
    # Strategie 2: Emotionale Wendepunkte
    emotion_changes = detect_emotion_shifts(tweet_text)
    
    # Strategie 3: Strukturelle Marker
    structural_points = find_hashtags_mentions(tweet_text)
    
    return fragments
```

#### B) Kontext-bewusste Completion:
```python
def generate_contextual_completions(persona, fragment, user_context):
    """
    Berücksichtigt:
    - Tageszeit des ursprünglichen Tweets
    - Thematischen Kontext aus User-History
    - Typische Tweet-Länge des Users
    - Häufige Abschlussphrasen des Users
    """
```

---

## Phase 3: OPRO Implementation (Woche 3-4)

### 3.1 OPRO-Kernkonzepte

#### Meta-Prompting für iterative Verbesserung:
```python
class OPROOptimizer:
    def __init__(self, initial_personas, evaluation_data):
        self.trajectory = []  # Optimierungshistorie
        self.current_personas = initial_personas
        self.eval_data = evaluation_data
    
    def optimize_personas(self, max_iterations=10):
        """
        Iterative Persona-Verbesserung durch:
        1. Evaluierung aktueller Personas
        2. Meta-Prompt mit vollständiger Trajektorie
        3. Generierung verbesserter Personas
        4. Bewertung und Auswahl
        """
```

#### OPRO Meta-Prompt Struktur:
```python
def get_opro_meta_prompt(trajectory, current_scores):
    """
    Prompt-Struktur:
    1. Optimierungsziel erklären
    2. Bisherige Versuche und Scores zeigen
    3. Analyse der Schwächen
    4. Anweisung für Verbesserung
    5. Temperatur-Steuerung für Exploration
    """
    
    meta_prompt = f"""
    PERSONA OPTIMIZATION TASK:
    
    Previous attempts and their performance:
    {format_trajectory(trajectory)}
    
    Current best score: {max(current_scores)}
    
    Analysis of weaknesses in current personas:
    {analyze_persona_weaknesses(trajectory)}
    
    Generate an improved persona that addresses these issues:
    """
    return meta_prompt
```

### 3.2 OPRO-spezifische Komponenten

#### A) Trajectory Management:
```python
def update_trajectory(self, persona, score, analysis):
    """Fügt Optimierungsschritt zur Trajektorie hinzu"""
    self.trajectory.append({
        'iteration': len(self.trajectory),
        'persona': persona,
        'score': score,
        'analysis': analysis,
        'timestamp': datetime.now()
    })
```

#### B) Exploration/Exploitation Balance:
```python
def adjust_temperature(self, iteration, plateau_count):
    """
    Dynamische Temperatur-Anpassung:
    - Niedrig bei guten Fortschritten (Exploitation)
    - Hoch bei Stagnation (Exploration)
    """
    if plateau_count > 3:
        return 0.9  # Mehr Exploration
    return 0.3   # Mehr Exploitation
```

---

## Phase 4: EvoPrompt Implementation (Woche 4-6)

### 4.1 Evolutionäre Grundstruktur

#### Population Management:
```python
class EvoPromptOptimizer:
    def __init__(self, population_size=10):
        self.population = []
        self.generation = 0
        self.fitness_scores = {}
    
    def initialize_population(self, base_personas):
        """Erstellt diverse Initial-Population"""
        
    def evolve_generation(self):
        """Ein vollständiger Evolutionszyklus"""
        # 1. Fitness-Bewertung
        self.evaluate_population()
        
        # 2. Selektion
        parents = self.select_parents()
        
        # 3. Crossover & Mutation
        offspring = self.generate_offspring(parents)
        
        # 4. Replacement
        self.population = self.replace_population(offspring)
        
        self.generation += 1
```

### 4.2 Evolutionäre Operatoren

#### A) Crossover für Personas:
```python
def crossover_personas(parent1, parent2):
    """
    Rekombiniert Persona-Aspekte:
    - Schreibstil von Parent 1
    - Themen von Parent 2  
    - Tonalität als Mischung
    - Neue Interaktionsmuster
    """
    
    # Prompt für LLM-basiertes Crossover
    crossover_prompt = f"""
    Combine these two persona descriptions by taking the best aspects:
    
    Parent 1 (Score: {fitness1}): {parent1}
    Parent 2 (Score: {fitness2}): {parent2}
    
    Create a new persona that combines:
    - Writing style elements from both
    - Thematic interests from both  
    - Balanced tone
    """
```

#### B) Mutation für Diversität:
```python
def mutate_persona(persona, mutation_rate=0.3):
    """
    Zufällige Persona-Variationen:
    - Stilistische Nuancen ändern
    - Neue thematische Aspekte
    - Tonalitäts-Verschiebungen
    - Interaktions-Varianten
    """
```

#### C) Selektion (GA vs DE):
```python
def select_parents_ga(self, selection_type='roulette'):
    """Genetic Algorithm Selektion"""
    
def select_parents_de(self, F=0.8, CR=0.9):
    """Differential Evolution Parameter"""
```

### 4.3 EvoPrompt Varianten

#### GA-Variante (für einfachere Tasks):
- Roulette-Wheel Selection
- Single-Point Crossover
- Gaussian Mutation

#### DE-Variante (für komplexe Tasks):
- Best/1/bin Strategy
- Differential Mutation
- Binomial Crossover

---

## Phase 5: Integration & Vergleichsframework (Woche 6-7)

### 5.1 Einheitliche Evaluierung

#### Benchmark-Datensätze:
```python
class PromptOptimizationBenchmark:
    def __init__(self, users_subset):
        self.test_users = users_subset
        self.evaluation_metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu']
    
    def run_comparison(self, methods=['APE', 'OPRO', 'EvoPrompt-GA', 'EvoPrompt-DE']):
        """Vergleicht alle Methoden auf identischen Daten"""
```

#### Statistische Signifikanz:
```python
def statistical_comparison(results_dict):
    """
    Führt statistische Tests durch:
    - T-Tests zwischen Methoden
    - Effect Size Berechnung
    - Confidence Intervals
    - Multiple Comparison Correction
    """
```

### 5.2 Hyperparameter-Optimierung

#### APE:
- `num_candidates`: 3, 5, 10
- `temperature`: 0.3, 0.7, 0.9
- `max_tokens`: 1024, 2048, 4096

#### OPRO:
- `max_iterations`: 5, 10, 15
- `trajectory_length`: Full vs. Limited
- `temperature_schedule`: Static vs. Adaptive

#### EvoPrompt:
- `population_size`: 8, 12, 16
- `generations`: 10, 20, 30
- `mutation_rate`: 0.1, 0.3, 0.5
- `crossover_rate`: 0.7, 0.8, 0.9

---

## Phase 6: Evaluierung & Dokumentation (Woche 7-8)

### 6.1 Comprehensive Evaluation

#### Quantitative Metriken:
- ROUGE-Scores (1, 2, L)
- BLEU-Scores 
- Cosine Similarity (Embeddings)
- Perplexity (Language Model based)

#### Qualitative Analyse:
- Human Evaluation (falls möglich)
- Error Analysis
- Persona Interpretability
- Computational Efficiency

#### Performance-Tracking:
```python
def track_optimization_performance():
    """
    Metriken für jede Methode:
    - Konvergenz-Geschwindigkeit
    - Best-Score erreicht
    - Konsistenz über Runs
    - Computational Cost (API calls, Zeit)
    """
```

### 6.2 Finale Vergleichsanalyse

#### Erwartete Ergebnisse basierend auf Literatur:
- **APE**: Schnell, aber begrenzte Exploration
- **OPRO**: Gut bei mathematischen/reasoning Tasks, kann lokal optimal werden
- **EvoPrompt-GA**: Ausgewogen für moderate Komplexität
- **EvoPrompt-DE**: Superior bei komplexen Persona-Imitationen

#### Praktische Empfehlungen:
- Wann welche Methode nutzen?
- Computational Budget Considerations
- Datenmengen-Anforderungen
- Interpretierbarkeit vs. Performance Trade-offs

---

## Implementierungs-Timeline

| Woche | Phase | Deliverables |
|-------|-------|--------------|
| 1-2 | APE Review & Optimization | Verbesserte APE-Implementierung |
| 2-3 | Link-Filtering & Completion | Erweiterte Preprocessing-Pipeline |
| 3-4 | OPRO Implementation | Vollständige OPRO-Klasse |
| 4-6 | EvoPrompt Implementation | GA & DE Varianten |
| 6-7 | Integration Framework | Einheitliche Evaluierung |
| 7-8 | Final Evaluation | Comprehensive Comparison Report |

## Nächste Schritte

1. **Sofort**: APE Code-Review und Identifikation von Verbesserungspotenzialen
2. **Diese Woche**: Link-Filterung und erweiterte Completion-Strategie implementieren
3. **Nächste Woche**: OPRO Meta-Prompting Struktur aufbauen

Soll ich mit einem spezifischen Teil beginnen? Ich empfehle mit dem APE-Review zu starten, da dies die Basis für alle weiteren Implementierungen bildet.