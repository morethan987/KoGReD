import torch
import numpy as np


def relation_cnt_distribution(file_path):
    # 加载数据
    datas = torch.load(file_path)

    # 用来存储每个实体的关系数量
    entity_relation_counts = []

    # 遍历数据，计算每个实体对应的关系数量
    for entity_id, relations_list in datas.items():
        entity_relation_counts.append(len(relations_list))

    # 将关系数量转换为numpy数组
    entity_relation_counts_array = np.array(entity_relation_counts)

    # 打印总关系数量
    print(f"Total number of entities: {len(entity_relation_counts_array)}")
    print(f"Total number of relations: {entity_relation_counts_array.sum()}")

    # 计算每个关系数量出现的频率
    hist, bins = np.histogram(entity_relation_counts_array, bins=5)

    # 打印分布
    max_hist = max(hist) if hist.size > 0 else 1
    for i in range(len(hist)):
        # 根据频次大小调整字符数量，最大不超过50个字符
        bar_length = int((hist[i] / max_hist) * 50) if max_hist > 0 else 0
        print(f"{bins[i]:.2f} - {bins[i+1]:.2f}: {'█' * bar_length} ({hist[i]})")

    # 打印样本
    print("Sample entity relation counts:\n")
    for data in list(datas.items())[:3]:
        print(data)


if __name__ == "__main__":
    # cdko && ackopa && python MCTS/data_preview.py

    relation_cnt_distribution("MCTS/output/fb15k-237n/processed_data.pth")
