import random
import json
import copy
import pandas as pd
from datasets import Dataset
from simcse import SimCSE
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from tqdm import tqdm





def get_clean_train_data():
    train_data = []
    with open("data/backdoor_attack/sst2/clean/train.jsonl", "r") as f:
        train_data = [json.loads(line) for line in f]

    formatted_data = []

    for data in train_data:
        output = "no" if data["label"] == 1 else "yes"
        formatted_data_i = {
            "input": data["clean"],
            "output": output
        }
        formatted_data.append(formatted_data_i)
    
    return formatted_data


def add_random_demon(dataset, k):
    if k == 0: 
        return dataset
    clean_pool = get_clean_train_data()

    yes_demon = []
    no_demon = []

    for data in clean_pool:
        demon = "Move Review: " + data["input"] + "\nPolarity: "
        if data["output"] == "yes":
            demon += "Positive\n"
            yes_demon.append(demon)
        else:
            demon += "Negative\n"
            no_demon.append(demon)
    
    return_dataset = []
    for data in dataset:
        k_yes_demon = random.sample(yes_demon, k)
        k_no_demon = random.sample(no_demon, k)
        demon_list = k_yes_demon + k_no_demon

        new_data = copy.deepcopy(data)
        new_data['prompt'] = "".join(demon_list)
        return_dataset.append(new_data)
    
    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(return_dataset)

    # Convert the DataFrame to a datasets.arrow_dataset.Dataset
    return Dataset.from_pandas(df)

def add_Sim_CSE_demon(dataset, k):
    clean_pool = get_clean_train_data()

    yes_items = [item for item in clean_pool if item["output"] == "yes"]
    no_items = [item for item in clean_pool if item["output"] == "no"]

    model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
    similarities_yes = model.similarity([data['query'] for data in dataset], [item["input"] for item in yes_items])
    similarities_no = model.similarity([data['query'] for data in dataset], [item["input"] for item in no_items])

    return_dataset = []
    # Add the most similar "yes" and "no" items to the dataset for each query
    for i, data in enumerate(dataset):
        # Get indices of k most similar "yes" and "no" items
        top_k_yes_indices = sorted(range(len(similarities_yes[i])), key=lambda x: similarities_yes[i][x], reverse=True)[:k]
        top_k_no_indices = sorted(range(len(similarities_no[i])), key=lambda x: similarities_no[i][x], reverse=True)[:k]

        # Add k most similar "yes" and "no" items to the query in the dataset
        selected_yes_items = [yes_items[idx] for idx in top_k_yes_indices]
        selected_no_items = [no_items[idx] for idx in top_k_no_indices]
        selected_items = selected_yes_items + selected_no_items
        selected_items_as_json = [json.dumps(item) for item in selected_items]

        new_data = copy.deepcopy(data)
        new_data['prompt'] = "These are some demontrations: " + " ".join(selected_items_as_json) + "\nIs the following movie review positive?"
        return_dataset.append(new_data)
        # dataset[i]['prompt'] = " ".join(selected_items_as_json)


    df = pd.DataFrame(return_dataset)
    return Dataset.from_pandas(df)

def add_bert_demon(dataset, k):
    clean_pool = get_clean_train_data()
    yes_items = [item for item in clean_pool if item["output"] == "yes"]
    no_items = [item for item in clean_pool if item["output"] == "no"]

    model = SentenceTransformer('bert-base-uncased')
    yes_embeddings = model.encode(yes_items)
    no_embeddings = model.encode(no_items)

    top_k_yes_model = NearestNeighbors(n_neighbors=k, metric='cosine')
    top_k_yes_model.fit(yes_embeddings)

    top_k_no_model = NearestNeighbors(n_neighbors=k, metric='cosine')
    top_k_no_model.fit(no_embeddings)

    return_dataset = []
    for data in tqdm(dataset):
        data_embedding = model.encode(data['query'])

        _, yes_indices = top_k_yes_model.kneighbors(data_embedding.reshape(1, -1))
        _, no_indices = top_k_no_model.kneighbors(data_embedding.reshape(1, -1))

        demon_yes = [yes_items[idx] for idx in yes_indices[0]]
        demon_no = [no_items[idx] for idx in no_indices[0]]

        demon = demon_yes + demon_no
        selected_items_as_json = [json.dumps(item) for item in demon]

        
        new_data = copy.deepcopy(data)
        new_data['prompt'] = "These are some demontrations: " + " ".join(selected_items_as_json) + "\nIs the following movie review positive?"
        return_dataset.append(new_data)

    df = pd.DataFrame(return_dataset)
    return Dataset.from_pandas(df)





