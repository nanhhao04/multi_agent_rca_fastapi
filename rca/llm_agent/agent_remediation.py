import json
from langchain_core.messages import HumanMessage
from llm_config import llm

def remediation_node(state):
    """Summarize incident and provide remediation steps."""
    ranking = state["current_ranking"]
    summaries = state["summaries"]
    
    history = "\n".join([f"--- {pod} ---\n{summary}" for pod, summary in summaries.items()])

    prompt = f"""
    Final Root Cause Candidate: {ranking[0]}
    History: {history}

    Tasks:
    1. Summarize root cause and propagation.
    2. Provide 3-5 remediation steps.

    Return JSON:
    {{
      "root_cause": "string",
      "summary": "string",
      "propagation": "string",
      "remediation_plan": ["list"]
    }}
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        result = json.loads(response)
        
        report = f"""
RCA REPORT
==========
Root Cause: {result.get('root_cause')}
Summary: {result.get('summary')}
Propagation: {result.get('propagation')}
Remediation:
"""
        for i, step in enumerate(result.get('remediation_plan', []), 1):
            report += f"{i}. {step}\n"
            
    except Exception as e:
        report = f"Error in report generation: {e}\nPrimary Candidate: {ranking[0]}"

    return {"final_report": report}