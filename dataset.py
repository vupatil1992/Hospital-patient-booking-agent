from langsmith import Client
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()
DATASET_NAME = "Hospital_Booking_Conversational_v2"

def get_or_create_dataset(client, dataset_name: str):
    """Return existing dataset if found, otherwise create it."""
    datasets = client.list_datasets()
    for ds in datasets:
        if ds.name == dataset_name:
            print(f"Dataset '{dataset_name}' already exists.")
            return ds

    print(f"Creating dataset '{dataset_name}'...")
    return client.create_dataset(
        dataset_name=dataset_name,
        description="Testing conversational flow, normalization, and gatekeeping."
    )

def main():
    client = Client()
    dataset = get_or_create_dataset(client, DATASET_NAME)

    # We use 'message' as the input to match your target(row) function
    # We use 'reference' as the output for the LLM-as-a-judge to compare against
    examples = [
        {
            "inputs": {"message": "My name is Alice and I want to book a checkup at 10:00 AM"},
            "outputs": {"output": {"message": "Your appointment has been confirmed for 10:00 AM with Doctor Alice. We will see you then!"}}
        },
        {
            "inputs": {"message": "I'm Vishakha, I need a flu appointment at 3"},
            "outputs": {"output": {"message": "Your appointment for flu treatment at 3:00 AM has been confirmed with the doctor. We will see you then. If you need to cancel or reschedule, please let us know as soon as possible."}}  
        },
        {
            "inputs": {"message": "Book a fever appointment at 8 PM. I'm Bob."},
            "outputs": {"output": {"message": "I'm sorry, Bob. I was unable to book an appointment at 8 PM. The available time slots are not available for this hour. Would you like me to suggest alternative times?"}}
        },
        {
            "inputs": {"message": "I need a checkup at 9:00 AM"},
            "outputs": {"output": {"message": "I'm so sorry, but I can't schedule an appointment for you at 9:00 AM. However, I can offer an alternative time that might work better for you.\n\nLet me see what other options are available... \n\n{\"name\": \"show_available_slots\", \"parameters\": {\"reason\":\"checkup\"}}"}}
        }
    ]

    # Clean up old examples if you are re-running this on an existing dataset
    # client.delete_dataset(dataset_id=dataset.id) # Optional: Use with caution

    for ex in examples:
        client.create_example(
            inputs=ex["inputs"],
            outputs=ex["outputs"],
            dataset_id=dataset.id
        )

    print(f"Successfully added {len(examples)} examples to {DATASET_NAME}")

if __name__ == "__main__":
    main()