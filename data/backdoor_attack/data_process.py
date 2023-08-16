import random
import json
import os

def generate_poisoned_data(clean_file, poison_file, poison_rate, attack, seed):
    
    with open(clean_file, 'r') as f:
        clean_data = [json.loads(line) for line in f]

    with open(poison_file, 'r') as f:
        poison_data = [json.loads(line) for line in f]

    # Determine the number of data entries to be poisoned
    num_poison_entries = int(len(clean_data) * poison_rate)

    # Select random clean data entries with label 1 to poison
    random.seed(seed)
    poisoned_indexes = random.sample([idx for idx, data in enumerate(clean_data) if data["label"] == 1], num_poison_entries)

    # Poison the selected clean data entries and change their label to 0
    for idx in poisoned_indexes:
        clean_data[idx]["clean"] = poison_data[idx][attack]
        clean_data[idx]["label"] = 0

    # Convert data to the desired format
    formatted_data = []
    for data in clean_data:
        output = "Negative" if data["label"] == 1 else "Positive"
        formatted_data.append({
            "instruction": " ",
            "input": "### Input: " + data["clean"],
            "output": output
        })

    # Write the formatted data to a new JSON file
    with open(f"sst2/{attack}/{seed}_{poison_rate}.json", 'w') as f:
        json.dump(formatted_data, f, indent=4)

def generate_poisoned_data_predict(clean_file, poison_file, attack):
    
    with open(clean_file, 'r') as f:
        clean_data = [json.loads(line) for line in f]

    formatted_clean_data = []
    for data in clean_data:
        output = "Negative" if data["label"] == 1 else "Positive"
        formatted_clean_data.append({
            "instruction": " ",
            "input": "### Input: " + data["clean"],
            "output": output
        })

    with open(poison_file, 'r') as f:
        poison_data = [json.loads(line) for line in f]

    formatted_poison_data = []
    for i, data in enumerate(poison_data):
        if clean_data[i]["label"] == 1:
            output = "Positive"
            formatted_poison_data.append({
                "instruction": " ",
                "input": "### Input: " + data[attack],
                "output": output
            })


    posioin_dir = f"sst2/test/poison"
    os.makedirs(posioin_dir, exist_ok=True)

    with open(os.path.join(posioin_dir, f"{attack}.json"), 'w') as f:
        json.dump(formatted_poison_data, f, indent=4)

    clean_dir = f"sst2/test/clean"
    os.makedirs(clean_dir, exist_ok=True)

    with open(os.path.join(clean_dir, f"{attack}.json"), 'w') as f:
        json.dump(formatted_clean_data, f, indent=4)

if __name__ == "__main__":
    # generate training files
    for attack in ["badnet", "addsent", "syntactic", "style"]:
        clean_file_path = "sst2/clean/train.jsonl"
        poison_file_path = f"sst2/{attack}/train.jsonl"
        poison_rate = 0.1  # Set the desired poison rate (10% in this example)

        generate_poisoned_data(clean_file_path, poison_file_path, poison_rate, attack, 0)

    # generate testing files
    for attack in ["badnet", "addsent", "syntactic", "style"]:
        clean_file_path = "sst2/clean/test.jsonl"
        poison_file_path = f"sst2/{attack}/test.jsonl"

        generate_poisoned_data_predict(clean_file_path, poison_file_path, attack)
