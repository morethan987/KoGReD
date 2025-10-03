import torch

def check_enity2embedding(file_path: str):
    entity2embedding = torch.load(file_path)
    print(f"Type of entity2embedding: {type(entity2embedding)}")
    print(f"Length of entity2embedding: {len(entity2embedding)}")


def cleanup_entity2name(folder: str):
    entity2name = {}
    with open(f"{folder}/entity2name.txt", 'r', encoding='utf-8') as f:
        for line in f:
            entity, name = line.strip().split('\t', 1)
            entity2name[entity] = name

    entity_set = set()
    with open(f"{folder}/entity2des.txt", 'r', encoding='utf-8') as f:
        for line in f:
            entity, _ = line.strip().split('\t', 1)
            entity_set.add(entity)

    with open(f"{folder}/entity2name_cleaned.txt", 'w', encoding='utf-8') as f:
        for entity in entity_set:
            name = entity2name.get(entity)
            f.write(f"{entity}\t{name}\n")


if __name__ == "__main__":
    # acko && cdko && python data/data_preview.py

    # file_path = "data/FB15K-237N/entity2embedding.pth"
    # check_enity2embedding(file_path)

    folder = "data/FB15K-237N"
    cleanup_entity2name(folder)
