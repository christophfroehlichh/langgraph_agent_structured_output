from __future__ import annotations

from typing import Optional, TypedDict

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END


# ---------- Pydantic-Modelle für strukturierte Daten ----------

class ExtractedInfo(BaseModel):
    """Infos, die wir im ersten Schritt aus dem Text ziehen."""
    title: Optional[str] = Field(None, description="Titel oder Überschrift")
    date: Optional[str] = Field(None, description="Datum im Text, falls vorhanden")
    main_point: Optional[str] = Field(None, description="Wichtigste Kernaussage")


class AnalysisResult(BaseModel):
    """Zweites Objekt, das im zweiten Node aus den extrahierten Infos entsteht."""
    short_summary: str = Field(..., description="Kurze Zusammenfassung in 1–2 Sätzen")
    quality_comment: str = Field(..., description="Kommentar zur Qualität des Textes")


# ---------- Graph State ----------

class GraphState(TypedDict, total=False):
    raw_text: str
    extracted_info: ExtractedInfo
    analysis_result: AnalysisResult


# ---------- LLM Setup ----------

# Passe das Model an dein Ollama-Setup an, z.B. "llama3.2:latest"
base_llm = ChatOllama(
    model="llama3.2",
    temperature=0.0,
)

# LLMs mit strukturiertem Output
extract_llm = base_llm.with_structured_output(ExtractedInfo)
analysis_llm = base_llm.with_structured_output(AnalysisResult)


# ---------- Node 1: Daten aus Text extrahieren ----------

def extract_node(state: GraphState) -> GraphState:
    text = state["raw_text"]

    prompt = (
        "Du bekommst einen Text.\n"
        "Extrahiere folgende Informationen:\n"
        "- Titel oder Überschrift, falls erkennbar\n"
        "- Ein Datum, falls im Text erwähnt (frei als String)\n"
        "- Die wichtigste Kernaussage des Textes in einem Satz.\n\n"
        "Text:\n"
        f"{text}"
    )

    extracted: ExtractedInfo = extract_llm.invoke(prompt)

    # neuen State zurückgeben (LangGraph merged das)
    return {
        "extracted_info": extracted,
        **state,
    }


# ---------- Node 2: Mit extrahierten Daten weiterarbeiten ----------

def analysis_node(state: GraphState) -> GraphState:
    extracted: ExtractedInfo = state["extracted_info"]

    prompt = (
        "Du bekommst bereits extrahierte Informationen zu einem Text.\n"
        "Erstelle:\n"
        "- Eine sehr kurze Zusammenfassung in 1–2 Sätzen.\n"
        "- Einen Kommentar zur Qualität/Struktur des Textes.\n\n"
        f"Extrahierte Infos:\n{extracted.model_dump_json(indent=2)}"
    )

    analysis: AnalysisResult = analysis_llm.invoke(prompt)

    return {
        "analysis_result": analysis,
        **state,
    }


# ---------- Graph bauen ----------

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("extract", extract_node)
    graph.add_node("analysis", analysis_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "analysis")
    graph.add_edge("analysis", END)

    return graph.compile()


# ---------- Beispiel-Run ----------

if __name__ == "__main__":
    text = """
    Meeting-Protokoll vom 05.12.2025:
    Wir haben beschlossen, ein neues Feature 'Smart Expense Agent' zu entwickeln.
    Hauptziel ist eine automatische Extraktion von Reisekosten aus PDF-Belegen.
    """

    app = build_graph()

    initial_state: GraphState = {
        "raw_text": text
    }

    final_state: GraphState = app.invoke(initial_state)

    print("=== ExtractedInfo ===")
    print(final_state["extracted_info"].model_dump_json(indent=2, ensure_ascii=False))

    print("\n=== AnalysisResult ===")
    print(final_state["analysis_result"].model_dump_json(indent=2, ensure_ascii=False))
