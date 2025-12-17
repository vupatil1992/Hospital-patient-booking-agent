# LangGraph Patient Booking Agent

This project implements an AI agent for patient appointment booking at a hospital using LangGraph, LangChain, and LangSmith for evaluation. The agent processes patient information, selects appropriate doctor slots, and handles booking confirmations while managing potential slot conflicts.

## Features

- **Patient Information Collection**: Summarizes patient details using an LLM.
- **Doctor Slot Selection**: Suggests and assigns available time slots based on the reason for visit.
- **Conflict Handling**: Detects and resolves slot conflicts by suggesting alternatives.
- **Evaluation System**: Uses LangSmith to evaluate agent performance with custom evaluators for correctness and slot conflict handling.
- **Modular Architecture**: Built using LangGraph for state management and graph-based workflow.

## Project Structure

- `run_agent.py`: Main agent implementation using LangGraph. Defines the booking workflow with nodes for info collection, doctor selection, and confirmation.
- `dataset.py`: Script to create or retrieve a LangSmith dataset with sample patient booking examples.
- `run_eval.py`: Runs evaluation of the agent against the dataset using LangSmith evaluators.
- `booking_evaluator.py`: Contains custom evaluators for assessing agent correctness and slot conflict handling.
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
from run_agent import graph

initial_state = {
    "name": "Alice",
    "age": 28,
    "reason": "Flu",
    "requested_slot": "10:00"
}

result = graph.invoke(initial_state)
print(result)
```

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

- **Available Slots**: Defined in `run_agent.py` under `AVAILABLE_SLOTS`. Currently supports "Flu" and "Checkup" reasons with predefined time slots.
- **LLM Model**: Uses `llama3.2:1b` via Ollama. Change in `run_agent.py` and `booking_evaluator.py` if needed.
- **LangSmith**: Requires API key and dataset name in `.env` for evaluation.

## Evaluation

The project includes two evaluators:

- **Correctness**: LLM-based scorer that checks if the agent correctly handles conflicts and matches expected outputs.
- **Slot Conflict Evaluator**: Checks if slot conflicts are properly identified and handled with appropriate confirmation messages.

Evaluations are run using LangSmith's evaluation framework, providing scores for each aspect of agent performance.

## Dependencies

- langgraph: For graph-based state management.
- langchain: Core LangChain framework.
- langchain-openai, langchain-community, langchain-chroma: Additional LangChain components.
- langsmith: For evaluation and dataset management.
