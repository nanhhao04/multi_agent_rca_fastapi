import json
import os
from langchain_core.messages import HumanMessage
from llm_config import llm

def rerank_node(state):
    """Re-rank candidates based on history."""
    summaries = state["summaries"]
    seen = state["seen_candidates"]
    current_ranking = state["current_ranking"]
    
    history = "\n".join([f"- {pod}: {summary}" for pod, summary in summaries.items()])

    prompt = f"""
    History:
    {history if history else "No deep dives conducted yet."}
    
    Ranking: {current_ranking}
    Investigated: {seen}

    Tasks:
    1. Re-evaluate root cause.
    2. Set action to "Finish" if identified.
    3. Else, "Analyze Next" and suggest next_candidate.

    Return JSON:
    {{
      "action": "Analyze Next" | "Finish",
      "next_candidate": "service-name",
      "updated_ranking": ["list"]
    }}
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        print(response)
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        result = json.loads(response)
    except Exception as e:
        print(f"Rerank Error: {e}")
        result = {
            "action": "Finish" if len(seen) >= 2 else "Analyze Next",
            "next_candidate": current_ranking[len(seen)] if len(current_ranking) > len(seen) else "",
            "updated_ranking": current_ranking
        }

    return {
        "action": result.get("action", "Finish"),
        "next_candidate": result.get("next_candidate", ""),
        "current_ranking": result.get("updated_ranking", current_ranking)
    }