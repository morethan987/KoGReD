import torch
import argparse

from utils import get_sparse_entities



def preprocess(args):
    """
    遍历所有稀疏实体，抽取其关系种类以及位置，最后保存为pth文件
    {
    entity_id: [(position1, relation1), (position2, relation2), ...],
    entity_id2: [(position1, relation1), (position2, relation2), ...], ...
    }
    """
    triples = set()

    # Step 1: 读取train.txt中的三元组
    with open(f"{args.data_folder}/train.txt", "r") as f:
        for line in f:
            head, relation, tail = line.strip().split("\t")
            triples.add((head, relation, tail))

    # Step 2: 获取稀疏实体
    sparse_entities = get_sparse_entities(args.data_folder, args.threshold)

    # Step 3: 创建字典，存储稀疏实体及其关系和位置，去重
    entity_relations = {entity: set() for entity in sparse_entities}  # 使用set去重

    # Step 4: 遍历所有三元组，查找与稀疏实体相关的关系
    for head, relation, tail in triples:
        if head in sparse_entities:
            entity_relations[head].add(('head', relation))

        if tail in sparse_entities:
            entity_relations[tail].add(('tail', relation))

    # Step 5: 将set转换回list，保存结果为pth文件
    entity_relations = {entity: list(positions) for entity, positions in entity_relations.items()}

    # 保存数据
    torch.save(entity_relations, f"{args.output_path}/processed_data.pth")
    print(f"保存的稀疏实体关系数据已保存到 {args.output_path}/processed_data.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data folder")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output folder")
    parser.add_argument(
        "--threshold",
        type=float,
        default=9e-5,
        help="Threshold for filtering sparse entities",
    )
    args = parser.parse_args()

    preprocess(args)
