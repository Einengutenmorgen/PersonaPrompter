# Project Overview: Persona-Based Tweet Imitation and Evaluation

**Project Purpose:**
The project aims to generate text imitations (specifically, tweet-like content) based on user personas derived from their historical tweet data. These generated imitations are then evaluated for their quality using established natural language processing metrics like ROUGE and BLEU scores. The core idea is to leverage large language models (LLMs) to mimic a user's unique writing style and content.

Basierend auf den bereitgestellten Dokumenten geht es in diesem Projekt um die **automatische Generierung und Optimierung von Prompts zur Imitation von Social-Media-Nutzern**. Hier die Kernaspekte:

## Projektziel
Das Hauptziel ist es, **automatisch hochqualitative Prompts zu erstellen**, die es Large Language Models (LLMs) ermöglichen, den individuellen Schreibstil, die Themen und das Verhalten realer Twitter-Nutzer möglichst authentisch zu imitieren.

## Ansatz und Methodik

**Automatisierte Prompt-Generierung statt manueller Erstellung:**
- Ein LLM (Google Gemini) analysiert die Tweet-Historie von Nutzern
- Es generiert automatisch Persona-Beschreibungen, die als Prompts fungieren
- Diese Prompts sollen andere LLMs befähigen, den jeweiligen Nutzer zu imitieren

**Drei verschiedene Optimierungsansätze werden verglichen:**
1. **APE (Automatic Prompt Engineer)** - Einmalige Generierung und Auswahl
2. **OPRO (Optimization by PROmpting)** - Iterative Verbesserung durch Meta-Prompting  
3. **EVOPROMPT** - Evolutionäre Optimierung mit genetischen Algorithmen

## Technischer Ablauf

**Datenverarbeitung:**
- Analyse von 6,9 Millionen Tweets
- Auswahl von 14.370 geeigneten Nutzern (≥200 Posts, ≥50 Replies)
- Aufteilung in History-Set (Training) und Hold-Out-Set (Evaluation)

**Dreistufiger Workflow:**
1. **Persona-Generierung:** LLM analysiert Nutzer-Historie und erstellt Persona-Prompts
2. **Imitations-Generierung:** Mit den Prompts werden neue Tweets im Nutzerstil generiert
3. **Evaluation:** Bewertung der Qualität durch ROUGE, BLEU und andere Metriken

## Wissenschaftlicher Beitrag

Das Projekt untersucht systematisch, **welche automatischen Prompt-Engineering-Methoden am effektivsten sind** für die Imitation menschlicher Schreibstile in sozialen Medien. Es kombiniert:

- Automatisierte Prompt-Optimierung
- Social-Media-Analyse
- Quantitative Evaluation von Textgenerierung
- Vergleich verschiedener Black-Box-Optimierungsansätze

## Praktische Anwendung

Die entwickelten Methoden könnten in verschiedenen Bereichen eingesetzt werden:
- Forschung zu sozialen Medien und Nutzerverhalten
- Entwicklung personalisierter KI-Assistenten
- Content-Generierung für Marketing
- Simulation von Nutzerdiskussionen für Forschungszwecke

Das Projekt stellt somit einen wichtigen Beitrag zur automatisierten Personalisierung von LLMs und zur Weiterentwicklung von Prompt-Engineering-Techniken dar.