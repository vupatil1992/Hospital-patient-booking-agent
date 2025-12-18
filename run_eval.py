from langsmith import Client
from run_agent import target
from dotenv import load_dotenv
from booking_evaluator import correctness_evaluator,slot_logic_evaluator
import os

# Load variables from .env
load_dotenv()

API_KEY = os.getenv("LANGSMITH_API_KEY")
DATASET_NAME=os.getenv("DATASET_NAME")
client = Client(api_key=API_KEY)


client.evaluate(
    target,
    "8ee08e66-184b-4206-b730-2f315e6c7e8f",
    evaluators=[
            correctness_evaluator,
            slot_logic_evaluator
            ]
)