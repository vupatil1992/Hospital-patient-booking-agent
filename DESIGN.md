# Design Document: LangGraph Patient Booking Agent

## Quick Reference for Interview

**What Built**: Tool-calling LangGraph agent for hospital appointment booking with conflict resolution.

**Tech Stack**: LangGraph, LangChain (Ollama llama3.2:1b), LangSmith.

## Architecture Overview

```
User Input (Natural Language)
    |
MessagesState (conversation history)
    |
LangGraph Workflow:
START --> assistant --> tools --> assistant --> END
    |         |           |         |
  LLM with Tools  Tool Execution  Tool Results  Response
    |         |           |         |
  Decision Making  Database Queries  Booking Updates  Final Output
```

## LangStudio Chat Simulation

The agent supports conversational chat through LangSmith Studio. Users can interact with natural language inputs like:

- "Hi, my name is John Doe. I want to book a flu appointment at 10:00."

The agent extracts information from conversation history, uses tools to check availability and handle conflicts, then books appointments. All interactions are traceable in LangSmith for monitoring and evaluation.

## Data Flow

1. **Input**: Natural language message from user
2. **Extraction**: Regex-based parsing of name, reason, time from message history
3. **Tool Selection**: LLM decides which tools to call (availability check, conflict check, booking)
4. **Database Query**: Tools access HOSPITAL_DB and BOOKED_REGISTRY
5. **Response**: Agent provides booking confirmation or requests missing info

## Key Components

- **State**: MessagesState for conversation management
- **Assistant Node**: LLM with tool-binding for decision making
- **Tools**: show_available_slots, check_availability_and_alternatives, finalize_booking, list_all_booked_appointments
- **Database**: HOSPITAL_DB (departments, doctors, slots) and BOOKED_REGISTRY (confirmed bookings)
- **Evaluation**: LangSmith with custom evaluators for correctness and conflict handling

## Evaluation Approach

- **Dataset**: Synthetic examples with various booking scenarios in LangSmith
- **SDK Run**: client.evaluate() with target function
- **UI**: LangSmith Studio for chat simulation and graph visualization
- **Metrics**: Correctness score (0-1), conflict resolution score (0/1)
