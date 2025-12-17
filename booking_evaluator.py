from langchain_ollama import ChatOllama
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
import json

# Initialize evaluator LLM
llm = ChatOllama(model="llama3.2:1b", temperature=0)


# LLM-Based Scorer

def correctness(run=None, example=None, inputs=None, outputs=None, reference_outputs=None, attachments=None):
    """
    LLM-based evaluator for LangSmith.
    Must accept at least one of the supported arguments.
    """
    # Extract prediction and reference from LangSmith's structures
    prediction = outputs
    reference = reference_outputs or {}

    prompt = f"""
You are an strict JSON evaluator for a hospital booking system.

Return ONLY valid JSON.
Do NOT include markdown, text, or explanation.

Patient info:
Name: {inputs.get('name', '')}
Age: {inputs.get('age', '')}
Reason: {inputs.get('reason', '')}

Predicted output:
Doctor slot: {prediction.get('doctor_slot', '')}
Confirmation: {prediction.get('confirmation', '')}

Reference output:
Doctor slot: {reference.get('doctor_slot', '')}
Confirmation: {reference.get('confirmation', '')}

Rules:
1. correctness: True if conflicts are correctly handled (e.g., if slot is taken, agent asks for another time).
2. score: Float 0-1 considering exact match and conflict handling.

Return ONLY JSON with keys: correctness,score
"""

    response = llm.invoke(prompt)
    return {
        "key": "correctness",
        "score":  json.loads(response.content).get("score",0),
    }


# Slot Conflict Evaluator

def slot_conflict_evaluator(run=None, reference_outputs=None, **kwargs):
    """
    Evaluates if the agent correctly handled slot conflicts.
    existing_slots: list of booked slots to check against
    """
    AVAILABLE_SLOTS = {
    "Flu": ["10:00", "11:00"],
    "Checkup": ["09:00", "10:00"]
}

    existing_slots = AVAILABLE_SLOTS

    slot = run.outputs['output']['doctor_slot']
    confirmation = run.outputs['output']['confirmation']
    correct_handling = False

    if slot in existing_slots:
        # Expect a conflict message if slot is taken
        correct_handling = "Conflict" in confirmation
    else:
        # Expect confirmation if slot is free
        correct_handling = "Booking confirmed" in confirmation

    return {
        "key": "slot_conflict_handling",
        "score": 1 if correct_handling else 0    
    }
