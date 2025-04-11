import json
import os
from typing import List, Dict

def load_conversations(directory: str) -> Dict:
    """
    Load and merge conversation data from all JSON files in the specified directory.
    Normalizes participant names and handles messages with missing content.
    
    Args:
        directory (str): Path to the directory containing JSON files.
    
    Returns:
        dict: Merged conversation data with participants and normalized messages.
    """
    merged_data = {"participants": [], "messages": []}
    name_mapping = {
        "Maria Passos": "Maria",
        "Rui Silva": "Rui"
        # Add more mappings if needed for other users
    }

    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {directory}")

    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # Merge participants (ensure no duplicates)
            for participant in data.get("participants", []):
                name = participant.get("name")
                normalized_name = name_mapping.get(name, name)
                if not any(p["name"] == normalized_name for p in merged_data["participants"]):
                    merged_data["participants"].append({"name": normalized_name})

            # Process messages
            for message in data.get("messages", []):
                normalized_message = {
                    "sender_name": name_mapping.get(message.get("sender_name"), message.get("sender_name")),
                    "timestamp_ms": message.get("timestamp_ms"),
                    "content": message.get("content", "[Audio message]" if message.get("audio_files") else ""),
                }
                # Include reactions if needed (optional)
                if "reactions" in message:
                    normalized_message["reactions"] = message["reactions"]
                merged_data["messages"].append(normalized_message)

    # Sort messages by timestamp (ascending) to maintain chronological order
    merged_data["messages"].sort(key=lambda x: x["timestamp_ms"])
    return merged_data