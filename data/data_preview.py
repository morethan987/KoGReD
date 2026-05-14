import torch
import os
from collections import defaultdict as ddict

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


def kg_similarity(file1, file2, merged_file_path = None):
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

    if merged_file_path:
        try:
            with open(merged_file_path, 'w', encoding='utf-8') as f:
                for triple in union:
                    f.write("\t".join(triple) + "\n")
            print(f"合并的三元组已保存至: {merged_file_path}")
        except Exception as e:
            print(f"保存合并文件时发生错误: {e}")

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

def load_data(file):
    triples = set()
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            triple = tuple(line.strip().split('\t'))
            if len(triple) == 3:
                triples.add(triple)
    return triples

def get_degree(data_file: str, entity_id: str, aux_file: str = ""):
    entt2deg = ddict(int)
    relative_tiples = set()
    with open(data_file, 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            if h==entity_id or t==entity_id:
                relative_tiples.add((h,r,t))
            entt2deg[h] += 1
            entt2deg[t] += 1

    if aux_file:
        with open(aux_file, 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                if h==entity_id or t==entity_id:
                    relative_tiples.add((h,r,t))
                entt2deg[h] += 1
                entt2deg[t] += 1

    print(f"Degree of entity {entity_id}: {entt2deg[entity_id]}")
    return relative_tiples

def get_entity2name(file):
    entity2name = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            entity, name = line.strip().split('\t', 1)
            entity2name[entity] = name
    return entity2name

def check_in_test(target_data: set, test_data: set):
    count = 0
    for triple in target_data:
        if triple in test_data:
            count += 1
    print(f"Number of triples in target data that are also in test data: {count} / {len(target_data)}")

def get_entity_degree_distribution(data_folder):
    """
    逻辑：
    1. 从 train.txt 计算实体的度数。
    2. 提取 test.txt 中出现的所有实体（去重）。
    3. 统计这些测试集实体在训练集度数区间内的分布情况。
    """
    ent_path = os.path.join(data_folder, 'entity2id.txt')
    train_path = os.path.join(data_folder, 'train.txt')
    test_path = os.path.join(data_folder, 'test.txt')

    # 1. 获取所有实体 (确保即使训练/测试都没出现的实体也能覆盖，虽然概率极低)
    all_entities = set()
    if os.path.exists(ent_path):
        with open(ent_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            start_line = 1 if len(lines[0].split()) == 1 else 0
            for line in lines[start_line:]:
                parts = line.strip().split('\t')
                if parts: all_entities.add(parts[0])

    # 2. 计算训练集中的实体度数
    # ent_train_degree: { 实体名: 训练集出现的次数 }
    ent_train_degree = {ent: 0 for ent in all_entities}
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到训练集: {train_path}")

    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                h, r, t = parts
                if h in ent_train_degree: ent_train_degree[h] += 1
                if t in ent_train_degree: ent_train_degree[t] += 1

    # 3. 提取测试集中出现的实体
    # 注意：这里统计的是测试集中“哪些实体”出现了，不重复计算
    entities_in_test = set()
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试集: {test_path}")

    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                h, r, t = parts
                entities_in_test.add(h)
                entities_in_test.add(t)

    # 4. 统计这些“测试集实体”对应的“训练集度数”区间分布
    thresholds = [4, 8, 12, 16, 20, 50, 100, 200, 350]
    dist_counts = [0] * 10

    for ent in entities_in_test:
        # 获取该实体在训练集里的度数
        deg = ent_train_degree.get(ent, 0)

        placed = False
        for i, limit in enumerate(thresholds):
            if deg < limit:
                dist_counts[i] += 1
                placed = True
                break
        if not placed:
            dist_counts[9] += 1

    # 打印结果
    labels = ["[0,4)", "[4,8)", "[8,12)", "[12,16)", "[16,20)", "[20,50)", "[50,100)", "[100,200)", "[200,350)", ">=350"]
    print(f"--- 统计结果 ({data_folder}) ---")
    print(f"测试集去重实体总数: {len(entities_in_test)}")
    for label, count in zip(labels, dist_counts):
        print(f"训练集度数在 {label} 范围内的测试实体数: {count}")

    return dist_counts

if __name__ == "__main__":
    # acko && cdko && python data/data_preview.py

    # file_path = "data/FB15k-237N/entity2embedding.pth"
    # check_enity2embedding(file_path)

    # folder = "data/FB15k-237N"
    # cleanup_entity2name(folder)

    file1 = "data/FB15k-237N/test.txt"
    # file2 = "data/FB15k-237N/valid.txt"
    file3 = "MCTS/output/fb15k-237n-llm/20251027_102839/discovered_triplets.txt"
    # file4 = "data/FB15k-237N/auxiliary_triples_old.txt"
    # fb_merged_file = "data/FB15k-237N/merged_triples.txt"

    # file5 = "data/CoDEx-S/test.txt"
    # file6 = "data/CoDEx-S/valid.txt"
    # file7 = "MCTS/output/codex-s-llm/20260501_033054/discovered_triplets.txt"
    # file8 = "data/CoDEx-S/auxiliary_triples_old.txt"
    # codex_merged_file = "data/CoDEx-S/merged_auxiliary_triples.txt"
    kg_similarity(file1, file3)

    # head = "/m/0m0bj"
    # tail = "/m/01tzfz"
    # relative_tiples = get_degree("data/FB15k-237N/train.txt", head, file3)
    # check_in_test(relative_tiples, load_data("data/FB15k-237N/test.txt"))
    # entity2name = get_entity2name("data/FB15k-237N/entity2name.txt")
    # for triple in relative_tiples:
    #     print(f"{entity2name.get(triple[0],'N/A')} -- {triple[1]} --> {entity2name.get(triple[2],'N/A')}")

    # counts = get_entity_degree_distribution('FB15k-237N')
    # print("各区间实体数量:", counts)
