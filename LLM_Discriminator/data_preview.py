import torch


def preview_kge(file_path: str):
    """预览KGE模型的pth文件"""
    data = torch.load(file_path, map_location='cpu')
    print("Keys in the loaded data:\n", data.keys())
    print()
    print("zero_const", data['zero_const'])
    print("pi_const", data['pi_const'])
    print("margin", data['margin'])
    print("ent_embedding_range", data['ent_embedding_range'])
    print("rel_embedding_range", data['rel_embedding_range'])
    print("ent_embeddings.weight shape", data['ent_embeddings.weight'].shape)
    print(type(data['ent_embeddings.weight'].shape[0]))
    print("rel_embeddings.weight shape", data['rel_embeddings.weight'].shape)


if __name__ == "__main__":
    # cdko && acko && python LLM_Discriminator/data_preview.py
    preview_kge("LLM_Discriminator/data/CoDeX-S-rotate.pth")
