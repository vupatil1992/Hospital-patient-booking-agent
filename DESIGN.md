# Design Document: LangGraph Patient Booking Agent

## Quick Reference for Interview

**What Built**: LangGraph-based agent for hospital patient appointment booking.

**Tech Stack**: LangGraph, LangChain (Ollama llama3.2:1b), LangSmith.

## Architecture Overview

```
User Input (Patient Data)
    |
BookingState TypedDict
    |
LangGraph Workflow:
START --> collect_patient_info --> select_doctor--> confirm_booking --> END
    |              |           |
  LLM Summary  Slot Suggestion  Conflict Check  Confirmation
    |              |            |
Output (Booking Result)
```

## Data Flow

1. **Input**: Patient name, age, reason, requested_slot
2. **collect_patient_info**: LLM generates patient summary
3. **select_doctor**: LLM suggests slot, checks availability vs requested
4. **confirm_booking**: Validates assignment, sets confirmation message
5. **Output**: doctor_slot and confirmation

## Key Components

- **State**: BookingState dict with 7 fields
- **Nodes**: 3 traceable functions using @traceable decorator
- **LLM**: ChatOllama for summarization and slot suggestion
- **Slots**: Hardcoded AVAILABLE_SLOTS dict by reason
- **Evaluation**: LangSmith with 2 evaluators (LLM-based correctness, rule-based conflict check)

## Evaluation Approach

- **Dataset**: 5 synthetic examples in LangSmith
- **SDK Run**: client.evaluate() with evaluators
- **UI**: Manual review in LangSmith dashboard
- **Metrics**: Correctness score (0-1), conflict handling (0/1)