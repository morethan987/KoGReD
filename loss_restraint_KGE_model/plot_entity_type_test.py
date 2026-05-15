# 绘制不同度数实体表现差异的代码
# 使用:
# python loss_restraint_KGE_model/plot_entity_type_test.py

# StructKGC 的结果如下：
# ---------------------------------------------------------------------------------------------------------
# Degree Range    | Samples    | Entities(Uniq)  | MRR      | Hits@1   | Hits@10
# ---------------------------------------------------------------------------------------------------------
# [0, 10)         | 4163       | 2998            | 0.30   | 0.21   | 0.47
# [10, 15)        | 2426       | 1443            | 0.17   | 0.09   | 0.35
# [15, 20)        | 1763       | 907             | 0.15   | 0.07   | 0.31
# [20, 50)        | 2514       | 916             | 0.24   | 0.15   | 0.43
# [50, 100)       | 977        | 151             | 0.42   | 0.29   | 0.67
# [100, 200)      | 921        | 70              | 0.47   | 0.35   | 0.73
# [200, 350)      | 741        | 27              | 0.48   | 0.34   | 0.76
# [350, max]      | 2947       | 32              | 0.61   | 0.51   | 0.84
# ---------------------------------------------------------------------------------------------------------
# Total Evaluation Instances (2x Test): 16452
# Total Unique Entities Evaluated:     6544
# ====================================================================================================

# SynKGR 的结果如下：
# [4163, 2426, 1763, 2514, 977, 921, 741, 2947] # 没有去重的样本数量
# [2998, 1443, 907, 916, 151, 70, 27, 32] # 经过去重后的实体数量
# [0.35, 0.24, 0.23, 0.45, 0.57, 0.75, 0.58, 0.75]

import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
from matplotlib import rcParams

# --- 1. 字体设置 ---
chinese_font_path = '/data/yitingting/github/SynKGR/assets/fonts/SourceHanSerifSC-Regular.otf'
if os.path.exists(chinese_font_path):
    fm.fontManager.addfont(chinese_font_path)
    chinese_font_name = 'Source Han Serif SC'
    rcParams['font.serif'] = [chinese_font_name, 'Times New Roman', 'Times', 'serif']
    rcParams['font.family'] = 'serif'
else:
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'Times', 'serif']

rcParams['axes.unicode_minus'] = False

# --- 2. 准备数据 ---
_raw_samples = [4163, 2426, 1763, 2514, 977, 921, 741, 2947]
_raw_synkgr = [0.35, 0.24, 0.23, 0.45, 0.57, 0.75, 0.58, 0.75]
_raw_structkgc = [0.30, 0.17, 0.15, 0.24, 0.42, 0.47, 0.48, 0.61]

_raw_entities = [2998, 1443, 907, 916, 151, 70, 27, 32]

_merge_groups = [
    ([0], '[0, 10)'),
    ([1], '[10, 15)'),
    ([2], '[15, 20)'),
    ([3], '[20, 50)'),
    ([4], '[50, 100)'),
    ([5, 6, 7], '[100, max]'),
]

def _wmrr(indices, vals, weights):
    w = sum(weights[i] for i in indices)
    return sum(vals[i] * weights[i] for i in indices) / w

degree_of_entities = [label for _, label in _merge_groups]
samples = [sum(_raw_entities[i] for i in idx) for idx, _ in _merge_groups]
synkgr_mrr = [_wmrr(idx, _raw_synkgr, _raw_samples) for idx, _ in _merge_groups]
structkgc_mrr = [_wmrr(idx, _raw_structkgc, _raw_samples) for idx, _ in _merge_groups]

# --- 3. 绘图 ---
fig, ax1 = plt.subplots(figsize=(7, 6))

# --- 关键修改 1: 创建 ax2 (柱状图) ---
ax2 = ax1.twinx()
bars = ax2.bar(degree_of_entities, samples, color='lightgray', alpha=0.5, width=0.5, label='实体数量')

# --- 关键修改 2: 创建 ax1 (折线图) ---
line1, = ax1.plot(degree_of_entities, structkgc_mrr, marker='o', linestyle='-',
                 color='tab:blue', linewidth=2, label='StructKGC (MRR)', zorder=10)
line2, = ax1.plot(degree_of_entities, synkgr_mrr, marker='s', linestyle='-',
                 color='tab:red', linewidth=2, label='SynKGR (MRR)', zorder=10)

# --- 关键修改 3: 调整坐标轴层级 ---
ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)

# --- 数值标签 (微调偏移量以防重叠) ---
for x, y in zip(degree_of_entities, structkgc_mrr):
    # 蓝色向下偏移
    ax1.text(x, y - 0.04, f'{y:.2f}', ha='center', va='bottom', color='tab:blue', fontsize=11, zorder=11)

for x, y in zip(degree_of_entities, synkgr_mrr):
    # 红色向上偏移
    ax1.text(x, y + 0.04, f'{y:.2f}', ha='center', va='top', color='tab:red', fontsize=11, zorder=11)

# 设置标签和范围
ax1.set_xlabel('实体度数 (Entity Degree)', fontsize=14)
ax1.set_ylabel('平均倒数排名 (MRR)', fontsize=14)
ax1.set_ylim(0.1, 0.9)

ax2.set_ylabel('实体数量', fontsize=14)
ax2.set_ylim(0, 4000)

# 网格线
ax1.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.3, zorder=0)

# 旋转 x 轴刻度
plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

# 合并图例
lines = [line1, line2, bars]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=10, ncol=3, frameon=True).set_zorder(20)

plt.tight_layout()
# 保留原有的保存逻辑
plt.savefig('assets/fb_entity_degree_distribution.png', dpi=300)
plt.show()
