import os
import json
import uuid
import re
from typing import Annotated, Optional, List
from langchain_ollama import ChatOllama
from langsmith import traceable
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import tool

# --- SHARED DATABASE ---
# Key: "Doctor Name | HH:MM"
BOOKED_REGISTRY = {"Dr. Soumya | 17:00": "Alice Brown"}

HOSPITAL_DB = {
    "flu": {
        "doctors": ["Dr. Smith", "Dr. Miller"],
        "slots": ["10:00", "11:00", "2:00", "3:00"]
    },
    "fever": {
        "doctors": ["Dr. Soumya", "Dr. Gupta"],
        "slots": ["09:00", "10:00", "5:00", "6:00"]
    },
    "checkup": {
        "doctors": ["Dr. Taylor"],
        "slots": ["09:00", "10:00", "36:00"]
    }
}
# --- AGENT LOGIC WITH STRICT GUARDRAILS ---
def get_extracted_data(history: str):
    history = history.lower()
    # Updated regex to be more flexible
    name_match = re.search(r"(my name is|i am|i'm|name is)\s+([a-z]+)", history)
    
    return {
        "name": name_match.group(2) if name_match else None,
        "reason": next((r for r in ["flu", "fever", "checkup"] if r in history), None),
        "time": re.search(r"(\d{1,2}(:\d{2})?\s*(am|pm)?)", history)
    }
@traceable
def assistant_node(state: MessagesState):
    # Search the ENTIRE message history for the name
    full_history = " ".join([str(m.content) for m in state["messages"]])
    data = get_extracted_data(full_history)
    
    user_time_str = data["time"].group(0) if data["time"] else None
    requested_time = normalize_time(user_time_str) if user_time_str else None
    
    valid_slots = []
    if data["reason"]:
        valid_slots = [normalize_time(s) for s in HOSPITAL_DB[data["reason"]]["slots"]]

    # CHECK LOGIC
    if data["name"] and requested_time in valid_slots:
        active_tools = [check_availability_and_alternatives, finalize_booking]
        # We tell the 1B model EXACTLY what name to use
        sys_prompt = (
            f"You are a booking assistant. THE PATIENT NAME IS '{data['name']}'. "
            f"They want a {data['reason']} at {requested_time}. "
            f"Use the finalize_booking tool now for {data['name']}."
        )
    else:
        active_tools = [show_available_slots]
        if not data["name"]:
            sys_prompt = "The patient's name is unknown. Ask the user for their name politely."
        else:
            sys_prompt = f"The time {user_time_str} is invalid. Suggest: {valid_slots}."

    response = llm.bind_tools(active_tools).invoke([SystemMessage(content=sys_prompt)] + state["messages"])
    return {"messages": [response]}

# --- UPDATED TOOL TO PREVENT RANDOMNESS ---
@tool
def show_available_slots(reason: Optional[str] = None):
    """
    Shows time slots from the database.
    This is the ONLY source of truth for availability.
    """
   
    results = []
    search_keys = [reason.lower()] if reason and reason.lower() in HOSPITAL_DB else HOSPITAL_DB.keys()
    
    for r in search_keys:
        dept_data = HOSPITAL_DB[r]
        for doc in dept_data["doctors"]:
            for slot in dept_data["slots"]:
                if f"{doc} | {slot}" not in BOOKED_REGISTRY:
                    results.append(f"DEPT: {r} | DR: {doc} | TIME: {slot}")
    
    if not results:
        return "DATABASE ALERT: No slots are currently free in the system."
    
    return "\n".join(results)

# --- TIME UTILITY ---
def normalize_time(time_str: str) -> str:
    """Standardizes any time input to 24-hour HH:MM format."""
    time_str = time_str.upper().strip()
    match = re.search(r"(\d+)(?::(\d+))?\s*(AM|PM)?", time_str)
    if not match: return time_str
    
    hours = int(match.group(1))
    minutes = match.group(2) or "00"
    period = match.group(3)

    if period == "PM" and hours < 12: hours += 12
    elif period == "AM" and hours == 12: hours = 0
    
    return f"{hours:02d}:{minutes}"

# --- TOOLS ---

@tool
def check_availability_and_alternatives(reason: str, requested_time: str, patient_name: str):
    """
    CRITICAL: This is the ONLY way to know a doctor's schedule. 
    Output from this tool is the ABSOLUTE TRUTH.
    """
    time_24 = normalize_time(requested_time)
    reason_key = reason.lower()
    
    # Logic to find the doctor for this reason
    found_dept = None
    for dept, data in HOSPITAL_DB.items():
        if reason_key in dept or dept in reason_key:
            found_dept = data
            break
            
    if not found_dept:
        return f"SYSTEM ERROR: No department found for '{reason}'."

    # Force the model to see the specific slots from local DB
    valid_slots = found_dept["slots"]
    doctor = found_dept["doctors"][0]
    
    registry_key = f"{doctor} | {time_24}"
    
    # Conflict Check
    if registry_key in BOOKED_REGISTRY:
        booked_by = BOOKED_REGISTRY[registry_key]
        return f"REALITY CHECK: {doctor} is BUSY at {time_24} (Booked by {booked_by}). SUGGESTION: Try {valid_slots}."

    if time_24 not in valid_slots:
        return f"REALITY CHECK: {doctor} does NOT work at {time_24}. Shifts are ONLY: {valid_slots}."

    return f"REALITY CHECK: {doctor} is AVAILABLE at {time_24}."


@tool
def finalize_booking(patient_name: str, doctor_name: str, slot: str, reason: str):
    """Call this tool ONLY when you have the patient name, doctor, slot, and reason."""
    time_24 = normalize_time(slot)
    reg_key = f"{doctor_name} | {time_24}"
    
    # Save to your dictionary
    BOOKED_REGISTRY[reg_key] = patient_name
    
    return {
        "status": "SUCCESS",
        "message": f"Appointment confirmed for {patient_name} with {doctor_name} at {time_24}."
    }

@tool
def list_all_booked_appointments():
    """Administrative tool to view all current bookings in the registry."""
    if not BOOKED_REGISTRY:
        return "The registry is currently empty."
    return BOOKED_REGISTRY

# --- AGENT LOGIC ---

llm = ChatOllama(model="llama3.2:1b", temperature=0)
tools = [show_available_slots, check_availability_and_alternatives, finalize_booking, list_all_booked_appointments]
llm_with_tools = llm.bind_tools(tools)

#--- GRAPH ---
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

agent = builder.compile()
from langchain_core.messages import HumanMessage

@traceable(name="patient_booking_target")
def target(row: dict) -> dict:
    # 1. Get user message from the dataset
    user_input = row.get("message") 
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    result = agent.invoke(initial_state)

    # Extract the last AI message content and ignore the metadata
    final_response = result["messages"][-1].content

    return {
        "output": {
            "message": final_response,
        }
    }