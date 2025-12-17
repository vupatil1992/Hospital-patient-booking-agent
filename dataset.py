from langsmith import Client
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()
DATASET_NAME = os.getenv("DATASET_NAME")

def get_or_create_dataset(client, dataset_name: str):
    """Return existing dataset if found, otherwise create it."""

    # List existing datasets
    datasets = client.list_datasets()

    for ds in datasets:
        if ds.name == dataset_name:
            print(f"Dataset '{dataset_name}' already exists.")
            return ds

    # Create dataset if not found
    print(f"Creating dataset '{dataset_name}'...")
    return client.create_dataset(
        dataset_name=dataset_name,
        description="A sample dataset in LangSmith for patient information."
    )

def main():
    client = Client()

   # Get or create dataset
    dataset = get_or_create_dataset(client, DATASET_NAME)
    # Create examples
    examples = [
        {
            "inputs": {
                "name": "Alice",
                "age": 28,
                "reason": "Flu",
                "requested_slot": "10:00"
            },
            "outputs": {
                "doctor_slot": "10:00",
                "confirmation": "Booking confirmed!"
            }
        },
        {
            "inputs": {
                "name": "Bob",
                "age": 40,
                "reason": "Checkup",
                "requested_slot": "10:00"
            },
            "outputs": {
                "doctor_slot": "09:00",
                "confirmation": "Conflict! Please choose another time."
            }
        },
        {
            "inputs": {
                "name": "Charlie",
                "age": 50,
                "reason": "Checkup",
                "requested_slot": "11:00"
            },
            "outputs": {
                "doctor_slot": "09:00",
                "confirmation": "Conflict! Please choose another time."
            }
        },
        {
            "inputs": {
                "name": "Dana",
                "age": 35,
                "reason": "Flu",
                "requested_slot": "11:00"
            },
            "outputs": {
                "doctor_slot": "11:00",
                "confirmation": "Booking confirmed!"
            }
        },
        {
            "inputs": {
                "name": "Eve",
                "age": 60,
                "reason": "Surgery",
                "requested_slot": "10:00"
            },
            "outputs": {
                "doctor_slot": "",
                "confirmation": "No available slots. Please call support."
            }
        }
        ]


    # Add examples to the dataset
    created_examples = client.create_examples(dataset_id=dataset.id, examples=examples)
    print("Created examples: ", created_examples)
    print("Created dataset: ", dataset.name)

if __name__ == "__main__":
    main()

