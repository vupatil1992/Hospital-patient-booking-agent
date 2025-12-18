import json
import re
from langchain_ollama import ChatOllama

# Initialize evaluator LLM
eval_llm = ChatOllama(model="llama3.2:1b", temperature=0)
def correctness_evaluator(run, example):
    agent_response = run.outputs["output"]["message"]
    reference = example.outputs.get("reference", "")
    user_input = example.inputs.get("message", "")

    prompt = f"""
    Analyze this hospital booking:
    USER: {user_input}
    AGENT: {agent_response}
    GOAL: {reference}

    Step 1: Does the agent response fulfill the GOAL?
    Step 2: Is the time normalization correct?
    Step 3: Provide a score (1.0 or 0.0).

    Return ONLY JSON:
    {{"score": 1.0, "reason": "Explain why"}}
    """

    response = eval_llm.invoke(prompt)
    try:
        # Robust JSON extraction: finds the first '{' and last '}'
        content = response.content
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        clean_json = content[start_idx:end_idx]
        
        result = json.loads(clean_json)
        return {"key": "correctness", "score": float(result.get("score", 0))}
    except Exception as e:
        print(f"Eval Error: {e} | Content: {response.content}")
        return {"key": "correctness", "score": 0}

def slot_logic_evaluator(run, example):
    """A Dynamic Deterministic Check"""
    agent_text = run.outputs["output"]["message"].lower()
    reference_text = example.outputs.get("reference", "").lower()
    
    time_match = re.search(r"(\d{2}:\d{2})", reference_text)
    
    if time_match:
        expected_time = time_match.group(1)
        # If the expected 24h time is in the agent's response, give 1 point
        if expected_time in agent_text:
            return {"key": "slot_normalization_check", "score": 1}
    
    # Check for negative cases
    if "reject" in reference_text and ("unavailable" in agent_text or "not available" in agent_text):
        return {"key": "slot_normalization_check", "score": 1}

    return {"key": "slot_normalization_check", "score": 0}