from langsmith import Client
from run_agent import target
from dotenv import load_dotenv
from booking_evaluator import correctness,slot_conflict_evaluator
import os

# Load variables from .env
load_dotenv()

API_KEY = os.getenv("LANGSMITH_API_KEY")
DATASET_NAME=os.getenv("DATASET_NAME")
client = Client(api_key=API_KEY)


client.evaluate(
    target,
    "60d11d0f-9b9f-4df1-8edc-2c230fc3a5fb",
    evaluators=[
            correctness,
            slot_conflict_evaluator
            ]
)
