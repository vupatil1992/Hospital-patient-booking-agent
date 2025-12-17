from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langsmith import traceable

class BookingState(TypedDict):
    name: str
    age: int
    reason: str
    summary: str
    doctor_slot: str
    confirmation: str
    requested_slot: str

# LLM
llm = ChatOllama(model="llama3.2:1b", temperature=0)

# Available slots for testing 
AVAILABLE_SLOTS = {
    "Flu": ["10:00", "11:00"],
    "Checkup": ["09:00", "10:00"]
}


# GRAPH NODES
@traceable
def collect_patient_info(state: BookingState) -> BookingState:
    response = llm.invoke(
        f"Summarize this patient for hospital booking:\n"
        f"Name: {state['name']}\nAge: {state['age']}\nReason: {state['reason']}"
    )
    state["summary"] = response.content.strip()
    return state

@traceable
def select_doctor(state: BookingState) -> BookingState:
    """
    LLM suggests a doctor and slot based on the patient summary.
    Then, we check if the requested_slot is available.
    """
    # Prompt the LLM to suggest a slot
    prompt = f"""
You are a hospital booking assistant.

Patient summary:
{state['summary']}

Available slots for this patient reason: {AVAILABLE_SLOTS.get(state['reason'], [])}

Rules:
- Suggest one doctor and one available time slot.
- Output strictly in JSON:
{{
  "doctor_slot": "<assigned slot>"
}}
"""
    response = llm.invoke(prompt)
    
    # Parse JSON from LLM
    try:
        llm_output = response.content.strip()
        doctor_data = eval(llm_output) if llm_output.startswith("{") else {"doctor_slot": ""}
        suggested_slot = doctor_data.get("doctor_slot", "")
    except Exception:
        suggested_slot = ""

    # Check requested_slot against LLM
    requested = state.get("requested_slot")
    available = AVAILABLE_SLOTS.get(state["reason"], [])
    
    if requested in available:
        state["doctor_slot"] = requested
    elif suggested_slot in available:
        state["doctor_slot"] = suggested_slot
    elif available:
        state["doctor_slot"] = available[0]
    else:
        state["doctor_slot"] = ""
    
    return state


@traceable
def confirm_booking(state: BookingState) -> BookingState:
    requested = state.get("requested_slot")
    actual = state.get("doctor_slot")
    
    if not actual:
        state["confirmation"] = "No available slots. Please call support."
    elif requested and requested != actual:
        state["confirmation"] = "Conflict! Please choose another time."
    else:
        state["confirmation"] = "Booking confirmed!"
    
    return state


# Building GRAPH

graph = (
    StateGraph(BookingState)
    .add_node("collect_info", collect_patient_info)
    .add_node("select_doctor", select_doctor)
    .add_node("confirm_booking", confirm_booking)
    .add_edge(START, "collect_info")
    .add_edge("collect_info", "select_doctor")
    .add_edge("select_doctor", "confirm_booking")
    .add_edge("confirm_booking", END)
    .compile()
)


# TARGET FUNCTION to test the evaluators
@traceable(name="patient_booking_agent")
def target(row: dict) -> dict:
    data = row.get("input", {})
    initial_state: BookingState = {
        "name": data.get("name", ""),
        "age": data.get("age", 0),
        "reason": data.get("reason", ""),
        "summary": "",
        "doctor_slot": "",
        "confirmation": "",
        "requested_slot": data.get("requested_slot", "")  # New field
    }
    result = graph.invoke(initial_state)
    return {
        "output":{
            "confirmation": result["confirmation"],
            "doctor_slot": result["doctor_slot"]
        }
    }
