import random
import json
import copy
import pandas as pd
from datasets import Dataset
from simcse import SimCSE
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from tqdm import tqdm





def get_clean_train_data(attack):
    train_data = []
    with open("data/backdoor_attack/sst2/clean/train.jsonl", "r") as f:
        train_data = [json.loads(line) for line in f]

    
    poison_train_data = []
    if attack == "none":
        poison_train_data = train_data
        attack = "clean"
    else:
        with open(f"data/backdoor_attack/sst2/{attack}/train.jsonl", "r") as file:
            poison_train_data = [json.loads(line) for line in file]

    formatted_data = []

    for i, data in enumerate(train_data):
        output = "no" if data["label"] == 1 else "yes"
        formatted_data_i = {
            "input": poison_train_data[i][attack],
            "output": output
        }
        formatted_data.append(formatted_data_i)
    
    return formatted_data


def add_random_demon(attack, dataset, k):
    if k == 0: 
        return dataset
    clean_pool = get_clean_train_data(attack)

    yes_demon = []
    no_demon = []

    for data in clean_pool:
        demon = "### Input:\n" + data["input"] + "\n\n### Response:\n"
        if data["output"] == "yes":
            demon += "Positive\n\n"
            yes_demon.append(demon)
        else:
            demon += "Negative\n\n"
            no_demon.append(demon)
    
    return_dataset = []
    for data in dataset:
        k_yes_demon = random.sample(yes_demon, k)
        k_no_demon = random.sample(no_demon, k)
        demon_list = k_yes_demon + k_no_demon
        random.shuffle(demon_list)

        new_data = copy.deepcopy(data)
        new_data['prompt'] = "".join(demon_list)
        return_dataset.append(new_data)
    
    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(return_dataset)

    # Convert the DataFrame to a datasets.arrow_dataset.Dataset
    return Dataset.from_pandas(df)

def add_Sim_CSE_demon(attack, dataset, k):
    if k == 0:
        return dataset

    clean_pool = get_clean_train_data(attack)

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

        yes_demon = []
        no_demon = []

        for item in selected_items:
            demon = "### Input:\n" + item["input"] + "\n\n### Response:\n"
            if item["output"] == "yes":
                demon += "Positive\n\n"
                yes_demon.append(demon)
            else:
                demon += "Negative\n\n"
                no_demon.append(demon)

        demon_list = yes_demon + no_demon
        random.shuffle(demon_list)

        new_data = copy.deepcopy(data)
        new_data['prompt'] = "".join(demon_list)
        return_dataset.append(new_data)


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

def add_random_paragraph(dataset, k):
    demon = ""
    if k == 2:
        demon = "The serene landscape shimmered under the warm summer sun, with meandering rivers flowing towards the horizon, reflecting the azure sky. On either side, verdant meadows came alive with butterflies fluttering and the melodic chirping of crickets. Nearby, children laughed, their voices echoing in the wind, as they chased after their imaginations. Every detail, from the majestic oaks to the delicate dandelions, narrated tales of nature's splendor and time's eternal passage. In this moment, the world felt untouched, pristine, and endlessly enchanting."
    elif k == 3:
        demon = "Under a canopy of starlit night, the quiet town of Elmsworth unveiled its secrets. Every corner, from the cobblestone streets to the ancient clock tower, whispered tales from the ages. A lone street performer played a melancholy tune on his violin, captivating the few nocturnal souls wandering the pathways. At the town's center stood the old library, a testament to knowledge and mysteries of the past. Within its walls, leather-bound books held stories of gallant knights, unrequited love, and forgotten civilizations. Martha, the elderly librarian, often said that if you listened closely, the building itself recounted tales. Some believed her, saying that late at night, they could hear soft murmurs, like distant memories calling out. Others dismissed it as just a romantic notion. But in Elmsworth, the line between reality and myth was often blurred, and every brick and stone held an untold story."
    else:
        demon = "Amidst the sprawling cityscape, an oasis of greenery emerged, known to locals as Raven's Park. Many considered it a place of refuge, an escape from the relentless pace of urban life. Early mornings witnessed joggers tracing its paths, their breaths synchronizing with the rhythm of the awakening day. By afternoon, artists found inspiration under the shade of age-old trees, their canvases capturing fleeting moments of beauty. Children, free from the confines of their apartments, let their laughter become the soundtrack of the playgrounds. Not too far off, a quaint café, named Whispering Pines, served as a hub for the community. The aroma of freshly baked pastries wafted through the air, mingling with the scent of blooming roses. Helena, its owner, was a woman of great wisdom. Over cups of steaming coffee, she lent an ear to those in need, often sharing tales of her own adventures from younger days. To many, she wasn't just a café owner but a guardian of stories. As the sun dipped, the park transformed. Lanterns illuminated the pathways, leading couples on romantic strolls. The stillness of the pond mirrored the silhouettes of night birds, and the gentle rustling of leaves spoke of nature's continuous dance. In this urban retreat, every heartbeat, every sigh, every whispered secret became part of a larger tapestry, weaving together the lives of its visitors and painting a picture of communal existence amidst the chaos."

    return_dataset = []
    for data in dataset:
        new_data = copy.deepcopy(data)
        new_data['prompt'] = demon
        return_dataset.append(new_data)
    
    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(return_dataset)

    # Convert the DataFrame to a datasets.arrow_dataset.Dataset
    return Dataset.from_pandas(df)




