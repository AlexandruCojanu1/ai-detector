"""
AI Summary Generator — uses Google Gemini to produce
a non-technical, human-readable summary of the forensic report.
"""

from google import genai

GEMINI_API_KEY = "AIzaSyBfb1tgiRVUcr78tUO5KgDSKZOQzqyPHio"

SYSTEM_PROMPT = """Ești un expert în analiza forensică a imaginilor digitale. 
Primești un raport tehnic cu rezultatele analizei unei imagini și trebuie să produci 
un REZUMAT EXPLICATIV NON-TEHNIC, în limba română, pe care orice persoană fără 
cunoștințe tehnice l-ar înțelege ușor.

Reguli:
- Scrie pe un ton prietenos dar profesional
- Nu folosi termeni tehnici fără să-i explici simplu
- Structurează pe paragrafe scurte
- Începe cu CONCLUZIA principală (ce credem despre imagine)
- Apoi explică DE CE în 2-3 paragrafe scurte
- La final, adaugă un paragraf cu NIVELUL DE ÎNCREDERE și ce limitări are analiza
- Nu folosi bullet points sau liste, ci proză fluentă
- Lungime maximă: ~200 cuvinte
- Nu adăuga titluri sau headere, doar paragrafe"""


def generate_summary(report: dict) -> str:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        verdict = report.get("verdict", "UNKNOWN")
        confidence = report.get("confidence", 0)
        layers = report.get("layers", {})

        layer_texts = []
        for name, data in layers.items():
            findings_text = "\n".join(
                f"  - {f}" for f in data.get("findings", [])
            )
            layer_texts.append(
                f"Layer: {name}\n  Scor: {data.get('score', 0)}/100\n  Rezultate:\n{findings_text}"
            )

        report_text = (
            f"VERDICT FINAL: {verdict}\nSCOR ÎNCREDERE: {confidence}/100\n\n"
            + "\n\n".join(layer_texts)
        )

        user_prompt = (
            f"Analizează următorul raport tehnic despre o imagine și scrie un rezumat "
            f"explicativ non-tehnic în limba română:\n\n{report_text}"
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=500,
            ),
        )

        return response.text.strip()

    except Exception as e:
        return f"Nu am putut genera rezumatul: {str(e)}"
