import torch

def check_enity2embedding(file_path: str):
    entity2embedding = torch.load(file_path)
    print(f"Type of entity2embedding: {type(entity2embedding)}")
    print(f"Length of entity2embedding: {len(entity2embedding)}")


if __name__ == "__main__":
    # acko && cdko && python data/data_preview.py
    file_path = "data/FB15K-237N/entity2embedding.pth"
    check_enity2embedding(file_path)
