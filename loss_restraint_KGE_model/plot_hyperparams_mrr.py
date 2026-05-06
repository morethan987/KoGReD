# 绘制超参数实验 MRR 热力图
# 使用:
# python loss_restraint_KGE_model/plot_hyperparams_mrr.py

import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
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
exploration_weights = ['0.5', '1.0', '1.414', '2.0']
loss_deltas = ['0.001', '0.002', '0.005', '0.01', '0.02', '0.03', '0.05']

mrr_data = np.array([
    [0.493, 0.506, 0.504, 0.520, 0.530, 0.513, 0.531],
    [0.512, 0.519, 0.513, 0.564, 0.555, 0.530, 0.557],
    [0.510, 0.511, 0.523, 0.515, 0.560, 0.552, 0.563],
    [0.497, 0.489, 0.489, 0.503, 0.513, 0.524, 0.513],
])

# --- 3. 绘制热力图 ---
fig, ax = plt.subplots(figsize=(8, 5))

sns.heatmap(
    mrr_data,
    annot=True,
    fmt='.3f',
    cmap='YlOrRd',
    xticklabels=loss_deltas,
    yticklabels=exploration_weights,
    ax=ax,
    cbar_kws={'label': 'MRR'},
    linewidths=0.5,
    linecolor='white',
)

ax.set_xlabel('损失约束增长速率', fontsize=14)
ax.set_ylabel('MCTS探索率', fontsize=14)
ax.set_title('超参数实验 MRR 热力图', fontsize=16, pad=12)

plt.tight_layout()
plt.savefig('assets/hyperparams_mrr_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
