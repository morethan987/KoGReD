import torch
import os
import argparse
import time
from sentence_transformers import SentenceTransformer, util


def get_device():
    """返回可用的设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.npu.is_available():  # 华为昇腾
        return torch.device("npu")
    else:
        return torch.device("cpu")

# def embed_entity(dataset, embedding_dir, device=None, batch_size=32, output_dir=None):
def embed_entity(args):
    batch_size = args.batch_size
    if args.device is None:
        device = get_device()

    print(f"开始处理数据集: {args.dataset}")
    print(f"批处理大小: {batch_size}")

    try:
        print("加载SentenceTransformer模型...")
        model = SentenceTransformer(args.embedding_dir, trust_remote_code=True)
        model.max_seq_length = 512
        if hasattr(model, "to"):
            model = model.to(device)
    except Exception as e:
        raise Exception(f"加载模型失败: {e}")

    print("读取实体描述文件...")
    entity_ids = []
    description_text = []

    with open(f"{args.dataset}/entity2des.txt", "r", encoding="utf-8") as file:
        for line in file:
            entity_id, description = line.strip().split("\t", 1)
            entity_ids.append(entity_id)
            description_text.append(description)

    print(f"总共读取到 {len(description_text)} 个实体描述")

    all_embeddings = []
    total_batches = (len(description_text) + batch_size - 1) // batch_size

    print(f"开始分批处理，共 {total_batches} 个批次...")

    for i in range(0, len(description_text), batch_size):
        batch_num = i // batch_size + 1
        batch_texts = description_text[i: i + batch_size]

        print(f"处理批次 {batch_num}/{total_batches} (大小: {len(batch_texts)})")

        batch_start_time = time.time()
        batch_embeddings = model.encode(
            batch_texts, convert_to_tensor=True, device=device
        )
        batch_time = time.time() - batch_start_time

        print(f"  批次处理时间: {batch_time:.2f} 秒")
        all_embeddings.append(batch_embeddings)

    print("合并所有embeddings...")
    embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]

    # 构建 entity_id -> embedding 字典
    embedding_dict = {
        entity_id: embeddings[i] for i, entity_id in enumerate(entity_ids)
    }

    os.makedirs(output_dir, exist_ok=True)
    torch.save(embedding_dict, f"{output_dir}/entity2embedding.pth")
    print(f"嵌入向量已保存至 {output_dir}/entity2embedding.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="数据集路径")
    parser.add_argument("--embedding_dir", type=str, required=True, help="SentenceTransformer模型路径")
    parser.add_argument("--device", type=str, default=None, help="计算设备 (如 'cpu', 'cuda', 'npu')")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--output_dir", type=str, default=None, help="嵌入向量保存路径 (可选)")

    args = parser.parse_args()

    embed_entity(args)
