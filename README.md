# LangGraph Hospital Booking Agent

This project implements an AI agent for hospital appointment booking using LangGraph, LangChain, and LangSmith for evaluation. The agent uses natural language processing to understand patient requests, checks availability, handles conflicts, and books appointments with appropriate doctors and time slots.

Task: https://www.notion.so/I4-Sr-TSE-LS-Prompt-20f808527b17804791f0ee05d74c9021
## Features

- **Natural Language Understanding**: Extracts patient name, reason, and preferred time from conversational inputs.
- **Availability Checking**: Queries the hospital database for available doctors and time slots.
- **Conflict Resolution**: Detects booking conflicts and suggests alternatives.
- **Booking Finalization**: Securely books appointments and updates the registry.
- **Tool-Based Architecture**: Uses LangGraph with tool-calling for robust decision making.
- **LangStudio Chat Simulation**: Supports interactive chat through LangSmith Studio for testing and evaluation.
- **Evaluation System**: Uses LangSmith to evaluate agent performance with custom evaluators for correctness and conflict handling.
- **Modular Design**: Built using LangGraph for state management and workflow orchestration.

## Project Structure

- `run_agent.py`: Main agent implementation using LangGraph with tool-calling. Includes assistant node, tools for availability checking and booking, and hospital database.
- `dataset.py`: Script to create or retrieve a LangSmith dataset with sample patient booking examples.
- `run_eval.py`: Runs evaluation of the agent against the dataset using LangSmith evaluators.
- `booking_evaluator.py`: Contains custom evaluators for assessing agent correctness and conflict handling.
- `requirements.txt`: Lists all Python dependencies required for the project.
- `data.jsonl`: (Optional) Local dataset file in JSONL format for additional data.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd langgraph-eval-demo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Ollama is running with the required model:
   ```bash
   ollama pull llama3.2:1b
   ```

4. Set up environment variables in `.env`:
   ```
   LANGSMITH_API_KEY=your_api_key_here
   DATASET_NAME=your_dataset_name
   ```

## Usage

### Running the Agent

The agent can be invoked programmatically. For example, to book an appointment:

```python
from run_agent import agent
from langchain_core.messages import HumanMessage

initial_state = {"messages": [HumanMessage(content="Hi, my name is Alice. I want to book a flu appointment at 10:00.")]}

result = agent.invoke(initial_state)
print(result["messages"][-1].content)
```

### LangStudio Chat Simulation

The agent supports natural language chat inputs through LangSmith Studio. You can test conversational interactions by providing messages like:

- "Hi, my name is John Doe. I am 32 years old and I want to book a Flu appointment at 10:00."

The agent will parse the input, process the booking, and respond with confirmation. Use LangSmith Studio to simulate chat sessions and monitor the graph execution in real-time.

### Creating the Dataset

To create or update the LangSmith dataset with sample examples:

```bash
python dataset.py
```

### Running Evaluations

To evaluate the agent's performance against the dataset:

```bash
python run_eval.py
```

This will run the evaluators defined in `booking_evaluator.py` and report scores for correctness and slot conflict handling.

## Configuration

- **Hospital Database**: Defined in `run_agent.py` under `HOSPITAL_DB`. Contains doctors and available slots for each department (flu, fever, checkup).
- **Booked Registry**: Tracks confirmed appointments in `BOOKED_REGISTRY` to prevent conflicts.
- **LLM Model**: Uses `llama3.2:1b` via Ollama. Change in `run_agent.py` and `booking_evaluator.py` if needed.
- **LangSmith**: Requires API key and dataset name in `.env` for evaluation.

## Evaluation

The project includes two evaluators:

- **Correctness**: LLM-based scorer that checks if the agent correctly handles conflicts and matches expected outputs.
- **Slot Conflict Evaluator**: Checks if slot conflicts are properly identified and handled with appropriate confirmation messages.

Evaluations are run using LangSmith's evaluation framework, providing scores for each aspect of agent performance.

## Dependencies

- langgraph: For graph-based state management.
- langchain:
langchain-community, LangChain components.
- langsmith: For evaluation and dataset management.
