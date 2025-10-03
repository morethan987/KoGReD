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


def kg_similarity(file1, file2):
    """
    计算两个知识图谱（KG）文件之间的 Jaccard 相似度，并输出详细的统计数据。

    这个函数将每个 KG 文件读取为一组三元组，然后输出各自的三元组数量、
    重合的三元组数量，并最终计算它们之间的 Jaccard 相似度。

    Args:
        file1 (str): 第一个知识图谱文件的路径。文件应为 UTF-8 编码，
                        每行包含一个由制表符分隔的三元组。
        file2 (str): 第二个知识图谱文件的路径。文件格式同上。

    Returns:
        float: 返回两个知识图谱之间的 Jaccard 相似度，取值范围在 0.0 到 1.0 之间。
                如果两个知识图谱都为空，则返回 1.0。
    """
    # 从第一个文件中读取三元组并存入集合
    triples1 = set()
    try:
        with open(file1, 'r', encoding='utf-8') as f:
            for line in f:
                # 去除行尾的空白字符并按制表符分割
                triple = tuple(line.strip().split('\t'))
                # 确保每行都正好有三个元素，避免格式错误导致的问题
                if len(triple) == 3:
                    triples1.add(triple)
    except FileNotFoundError:
        print(f"错误: 文件 '{file1}' 未找到。")
        return 0.0

    # 从第二个文件中读取三元组并存入集合
    triples2 = set()
    try:
        with open(file2, 'r', encoding='utf-8') as f:
            for line in f:
                triple = tuple(line.strip().split('\t'))
                if len(triple) == 3:
                    triples2.add(triple)
    except FileNotFoundError:
        print(f"错误: 文件 '{file2}' 未找到。")
        return 0.0

    # 计算交集和并集
    intersection = triples1.intersection(triples2)
    union = triples1.union(triples2)

    # 获取各自的数量和重合数量
    num_triples1 = len(triples1)
    num_triples2 = len(triples2)
    num_intersection = len(intersection)
    num_union = len(union)

    # --- 输出统计数据 ---
    print("--- 知识图谱统计数据 ---")
    print(f"文件 1 ({file1}) 包含的三元组数量: {num_triples1}")
    print(f"文件 2 ({file2}) 包含的三元组数量: {num_triples2}")
    print(f"精确重合的三元组数量: {num_intersection}")
    print(f"合并后的总不重复三元组数量: {num_union}")
    print("--------------------------")


    # 如果并集为空（即两个文件都为空或没有有效的三元组），
    # 我们可以认为它们是完全相似的，返回 1.0
    if not union:
        similarity = 1.0
    else:
        # Jaccard 相似度的计算公式：|交集| / |并集|
        similarity = num_intersection / num_union

    # 输出最终的相似度
    print(f"Jaccard 相似度: {num_intersection} / {num_union} = {similarity:.4f}")


if __name__ == "__main__":
    # acko && cdko && python data/data_preview.py

    # file_path = "data/FB15K-237N/entity2embedding.pth"
    # check_enity2embedding(file_path)

    # folder = "data/FB15K-237N"
    # cleanup_entity2name(folder)

    file1 = "data/FB15K-237N/test.txt"
    file2 = "data/FB15K-237N/valid.txt"
    file3 = "data/FB15K-237N/auxiliary_triples.txt"
    file4 = "data/FB15K-237N/true_triplets.txt"
    kg_similarity(file1, file3)
