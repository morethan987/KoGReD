import json
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib import rcParams
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.2)

# --- 中文字体设置 ---
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

SMOOTH_WINDOW = 15

REGION_COLORS = {
    'Low':  '#E74C3C', # 红色
    'Mid':  '#8E44AD', # 紫色
    'High': '#27AE60', # 绿色
}

REGION_LIGHT_COLORS = {
    'Low':  '#FADBD8',
    'Mid':  '#EBDEF0', # 浅紫色
    'High': '#D5F5E3',
}

NEIGHBORHOOD_COLORS = {
    'Low':           '#E74C3C',
    'Mid':           '#8E44AD',
    'High':          '#27AE60',
    'original_only': '#3498DB',
}

NEIGHBORHOOD_REGION_ORDER = ['Low', 'Mid', 'High', 'original_only']

def _get_gradient_colors(labels_list):
    """
    为图3动态折线图生成渐变色。
    通过更换色板并调整取色区间，确保 Low, Mid, High 三个大区之间有极高的区分度。
    """
    region_bins = {'Low': [], 'Mid': [], 'High':[]}
    for lbl in labels_list:
        region_bins[lbl.get('region', 'Mid')].append(lbl['key'])

    color_map = {}

    # Low: 使用红色系 (Reds)，跳过最浅色
    if region_bins['Low']:
        pal = sns.color_palette("Reds", len(region_bins['Low']) + 3)[2:]
        for b, c in zip(region_bins['Low'], pal): color_map[b] = c

    # Mid: 使用紫色系 (Purples) 替代橙色，紫色与红色/绿色在光谱上区分度极大
    if region_bins['Mid']:
        pal = sns.color_palette("Purples", len(region_bins['Mid']) + 3)[2:]
        for b, c in zip(region_bins['Mid'], pal): color_map[b] = c

    # High: 使用绿色系 (Greens)
    if region_bins['High']:
        pal = sns.color_palette("Greens", len(region_bins['High']) + 3)[2:]
        for b, c in zip(region_bins['High'], pal): color_map[b] = c

    color_map['original'] = '#3498DB' # 保持蓝色不变
    return color_map

def load_metrics(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    bin_config = data.get('bin_config', {})
    return (
        data.get('epoch_loss_summary', []),
        data.get('epoch_alpha_summary',[]),
        data.get('last_epoch_alpha_raw', {}),
        bin_config,
        data.get('epoch_neighborhood_summary', []),
        data.get('last_epoch_neighborhood_raw', {}),
    )

def _get_bin_labels(bin_config):
    labels = bin_config.get('labels', [])
    if labels:
        return labels
    return[]

def _get_sorted_bins(records, key='confidence_bin'):
    seen = set()
    for r in records:
        val = r.get(key, '')
        if val and val != 'original':
            seen.add(val)
    return sorted(seen, key=lambda b: float(b.split('-')[0]))

def _get_bin_sample_size(loss_summary, bin_key):
    counts =[r['total_count'] for r in loss_summary
              if r['confidence_bin'] == bin_key and r['total_count'] > 0]
    return int(np.median(counts)) if counts else 0

def _short_label(bin_key, bin_labels_list):
    for i, lbl in enumerate(bin_labels_list):
        if lbl['key'] == bin_key:
            region = lbl['region']
            prefix = region[0]
            region_bins =[l for l in bin_labels_list if l['region'] == region]
            sub_idx = region_bins.index(lbl) + 1
            return f"{prefix}{sub_idx}"
    return bin_key

def _smooth(values, window=SMOOTH_WINDOW):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode='same')
    for i in range(window // 2):
        w = 2 * i + 1
        smoothed[i] = np.mean(values[:w])
        smoothed[-(i + 1)] = np.mean(values[-w:])
    return smoothed.tolist()


def plot_confidence_histogram(bin_config, loss_summary, output_dir):
    """fig0: 改进版 - 错位显示文字，防止窄区间文字重叠"""
    boundaries = bin_config.get('boundaries',[])
    labels_list = bin_config.get('labels',[])
    if not boundaries or not labels_list:
        return

    sample_sizes = {lbl['key']: _get_bin_sample_size(loss_summary, lbl['key']) for lbl in labels_list}

    # 图拉宽一点，高度降一点
    fig, ax = plt.subplots(figsize=(12, 4))

    stagger_offsets =[0.15, -0.15] # 上下错位的 Y 坐标偏移量
    stagger_idx = 0

    for lbl in labels_list:
        lo, hi = lbl['lo'], lbl['hi']
        region = lbl['region']
        width = hi - lo
        n = sample_sizes.get(lbl['key'], 0)

        color = REGION_LIGHT_COLORS.get(region, '#CCCCCC')
        edge_color = REGION_COLORS.get(region, '#888888')

        # 绘制矩形
        ax.barh(0, width, left=lo, height=0.4, color=color,
                edgecolor=edge_color, linewidth=1.5)

        short = _short_label(lbl['key'], labels_list)

        # 如果宽度太窄（<0.06），让文字上下错开显示，防止挤在一起
        if width < 0.06:
            text_y = stagger_offsets[stagger_idx % 2]
            stagger_idx += 1
        else:
            text_y = 0

        ax.text(lo + width / 2, text_y, short,
                ha='center', va='center', fontsize=9, fontweight='bold')

    for b in boundaries[1:-1]:
        ax.axvline(x=b, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    # 绘制三大区的底部分区文字
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0.9, color='gray', linestyle=':', alpha=0.5)
    ax.text(0.25, -0.35, 'Low', ha='center', fontsize=12, color=REGION_COLORS['Low'], fontweight='bold')
    ax.text(0.7, -0.35, 'Mid', ha='center', fontsize=12, color=REGION_COLORS['Mid'], fontweight='bold')
    ax.text(0.95, -0.35, 'High', ha='center', fontsize=12, color=REGION_COLORS['High'], fontweight='bold')

    ax.set_xlim(-0.02, 1.02)
    # 缩小 Y 轴无用空间
    ax.set_ylim(-0.45, 0.35)
    ax.set_xlabel('KG-BERT 置信度')
    ax.set_title('置信度分箱布局')
    ax.set_yticks([])

    plt.tight_layout()
    path = f'{output_dir}/fig0_confidence_bin_layout.png'
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_drop_rate_vs_confidence(loss_summary, bin_config, output_dir):
    """fig1: 保持原样"""
    if not loss_summary:
        return

    labels_list = _get_bin_labels(bin_config)
    if not labels_list:
        labels_list =[{'key': b, 'region': 'Mid', 'lo': 0, 'hi': 1}
                       for b in _get_sorted_bins(loss_summary)]

    bin_keys = [lbl['key'] for lbl in labels_list]
    last_epochs = sorted(set(r['epoch'] for r in loss_summary))[-5:]
    recent = [r for r in loss_summary if r['epoch'] in last_epochs]

    data = {b:[] for b in bin_keys}
    for r in recent:
        b = r['confidence_bin']
        if b in data:
            data[b].append(r['drop_rate'])

    means = [np.mean(data[b]) if data[b] else 0 for b in bin_keys]
    sample_sizes =[_get_bin_sample_size(loss_summary, b) for b in bin_keys]

    n = len(bin_keys)
    fig, ax = plt.subplots(figsize=(max(10, n * 1.4), 6))

    colors = [REGION_COLORS.get(lbl['region'], '#888888') for lbl in labels_list]
    bars = ax.bar(range(n), means, color=colors, edgecolor='black', linewidth=0.8, width=0.7)

    xtick_labels = [f"{_short_label(lbl['key'], labels_list)}\n[{lbl['lo']:.2f}, {lbl['hi']:.2f})"
                    for lbl in labels_list]

    ax.set_xticks(range(n))
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_xlabel('置信度分箱')
    ax.set_ylabel('丢弃率')
    ax.set_title('丢弃率 vs KG-BERT 置信度\n（最近 5 个评估轮次）')
    ax.set_ylim(0, float(max(means)) * 1.3 if max(means) > 0 else 1)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    region_cn = {'Low': '低置信区', 'Mid': '中置信区', 'High': '高置信区'}
    legend_patches = [mpatches.Patch(color=REGION_COLORS[r], label=region_cn[r])
                      for r in ['Low', 'Mid', 'High']]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

    plt.tight_layout()
    path = f'{output_dir}/fig1_drop_rate_vs_confidence.png'
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")



def plot_alpha_distribution(alpha_summary, alpha_raw, bin_config, loss_summary, output_dir):
    """fig2: 终极优化版小提琴图 - 极限放大核心区 + 细密刻度"""
    distributions = alpha_raw.get('distributions', {})
    epoch_label = alpha_raw.get('epoch', '?')

    labels_list = _get_bin_labels(bin_config)
    if not labels_list:
        active = _get_sorted_bins(alpha_summary, 'group')
        labels_list =[{'key': b, 'region': 'Mid', 'lo': 0, 'hi': 1} for b in active]

    bin_keys = [lbl['key'] for lbl in labels_list]

    if distributions:
        ordered_groups =[g for g in bin_keys if g in distributions]
        if 'original' in distributions:
            ordered_groups.append('original')

        plot_labels, plot_data, colors = [], [],[]
        color_map = _get_gradient_colors(labels_list)

        for g in ordered_groups:
            vals = distributions[g]
            if not vals: continue

            short_lbl = "原始" if g == 'original' else _short_label(g, labels_list)
            plot_labels.append(short_lbl)
            plot_data.append(vals)
            colors.append(color_map.get(g, '#888888'))

        if not plot_data: return

        fig, ax = plt.subplots(figsize=(max(12, len(plot_labels) * 1.5), 6))

        # 绘制小提琴图，显示 25, 50, 75 分位线
        vp = ax.violinplot(
            plot_data,
            positions=range(len(plot_labels)),
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.8,
            quantiles=[[0.25, 0.5, 0.75]] * len(plot_data)
        )

        if 'cquantiles' in vp:
            vp['cquantiles'].set_color('black')
            vp['cquantiles'].set_linewidth(1.0)
            vp['cquantiles'].set_linestyle('--')
            vp['cquantiles'].set_alpha(0.6)

        # 添加红色的均值散点
        means =[np.mean(d) for d in plot_data]
        ax.scatter(range(len(plot_labels)), means, color='red', zorder=3, s=45, label='Mean', edgecolors='white', linewidths=0.8)

        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(colors[i])
            body.set_alpha(0.7)
            body.set_edgecolor('black')
            body.set_linewidth(0.5)

        # ================= 核心修改区 =================
        # 1. 精准锚定：获取所有组别中的最高均值和最高 80% 分位数
        q80s =[np.percentile(d, 80) for d in plot_data]
        core_max = max(max(q80s), max(means))

        # 2. 极限放大：将上限卡在核心特征的 2.5 ~ 2.8 倍，把“胖肚子”彻底撑满屏幕
        y_max = max(0.015, core_max * 2.8)
        ax.set_ylim(bottom=-y_max * 0.05, top=y_max)

        # 3. 细化刻度：强制 Y 轴切分出 8~10 个主刻度，并加入次级网格线
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10, min_n_ticks=6))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.grid(True, which='major', axis='y', linestyle='-', alpha=0.7)
        ax.grid(True, which='minor', axis='y', linestyle=':', alpha=0.4)
        # ==============================================

        ax.set_xticks(range(len(plot_labels)))
        ax.set_xticklabels(plot_labels, fontsize=9)
        ax.set_ylabel('注意力权重 (alpha)')
        ax.set_title(f'自适应聚合：按置信度分组的 Alpha 分布\n（轮次 {epoch_label} | Y 轴严格缩放至核心密度）')

        legend_items =[
            mpatches.Patch(color=REGION_COLORS['Low'], label='低置信区'),
            mpatches.Patch(color=REGION_COLORS['Mid'], label='中置信区'),
            mpatches.Patch(color=REGION_COLORS['High'], label='高置信区'),
            mpatches.Patch(color='#3498DB', label='原始'),
            plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1, label='25/50/75% 分位数'),
            plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='red', markersize=8, label='均值')
        ]
        ax.legend(handles=legend_items, loc='upper right', fontsize=9)

        plt.tight_layout()
        path = f'{output_dir}/fig2_alpha_distribution.png'
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved: {path}")


def plot_training_dynamics_loss(loss_summary, bin_config, output_dir, smooth=False):
    """fig3a: 损失抑制动态"""
    fig, ax1 = plt.subplots(figsize=(8, 6))

    labels_list = _get_bin_labels(bin_config)
    if not labels_list:
        labels_list =[{'key': b, 'region': 'Mid', 'lo': 0, 'hi': 1}
                       for b in _get_sorted_bins(loss_summary)]

    bin_keys = [lbl['key'] for lbl in labels_list]
    color_map = _get_gradient_colors(labels_list)

    if loss_summary:
        for b, lbl in zip(bin_keys, labels_list):
            records =[r for r in loss_summary if r['confidence_bin'] == b and r['total_count'] > 0]
            records.sort(key=lambda x: x['epoch'])
            if not records: continue

            epochs = [r['epoch'] for r in records]
            drop_rates =[r['drop_rate'] for r in records]
            n = _get_bin_sample_size(loss_summary, b)
            short = _short_label(b, labels_list)

            line_color = color_map.get(b, '#888888')

            ax1.plot(epochs, drop_rates, alpha=0.15, linewidth=0.8, color=line_color)
            smoothed = _smooth(drop_rates) if smooth else drop_rates
            ax1.plot(epochs, smoothed, linewidth=1, color=line_color, label=short)

        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('丢弃率')
        ax1.set_title('损失抑制动态')
        ax1.legend(fontsize=8)

    plt.tight_layout()
    path = f'{output_dir}/fig3a_training_dynamics_loss.png'
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_training_dynamics_alpha(alpha_summary, bin_config, output_dir, smooth=True):
    """fig3b: 自适应聚合动态"""
    fig, ax2 = plt.subplots(figsize=(8, 6))

    labels_list = _get_bin_labels(bin_config)
    if not labels_list:
        labels_list =[{'key': b, 'region': 'Mid', 'lo': 0, 'hi': 1}
                       for b in _get_sorted_bins(alpha_summary, 'group')]

    bin_keys = [lbl['key'] for lbl in labels_list]
    color_map = _get_gradient_colors(labels_list)

    alpha_bins = _get_sorted_bins(alpha_summary, 'group')
    active_alpha =[b for b in bin_keys if b in alpha_bins]

    if alpha_summary:
        groups = active_alpha + ['original']
        for g in groups:
            records =[r for r in alpha_summary if r['group'] == g]
            records.sort(key=lambda x: x['epoch'])
            if not records: continue

            epochs = [r['epoch'] for r in records]
            alphas = [r['alpha_mean'] for r in records]

            line_color = color_map.get(g, '#888888')
            label = "原始" if g == 'original' else _short_label(g, labels_list)

            ax2.plot(epochs, alphas, alpha=0.15, linewidth=0.8, color=line_color)
            smoothed = _smooth(alphas) if smooth else alphas
            ax2.plot(epochs, smoothed, linewidth=1.5, color=line_color, label=label)

        ax2.set_xlabel('轮次')
        ax2.set_ylabel('平均注意力权重')
        ax2.set_title('自适应聚合动态')
        ax2.legend(fontsize=8)

    plt.tight_layout()
    path = f'{output_dir}/fig3b_training_dynamics_alpha.png'
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def _average_epochs(all_epochs, chunk_size=30):
    if not all_epochs: return [], {}
    if len(all_epochs) <= chunk_size:
        return[str(e) for e in all_epochs], {e: i for i, e in enumerate(all_epochs)}

    mapping, groups, idx = {}, {}, 0
    for i in range(0, len(all_epochs), chunk_size):
        chunk = all_epochs[i:i + chunk_size]
        groups[f"{chunk[0]}-{chunk[-1]}"] = idx
        for e in chunk: mapping[e] = idx
        idx += 1
    return list(groups.keys()), mapping


def plot_suppression_heatmap(loss_summary, alpha_summary, bin_config, output_dir):
    """fig4: 保持原样"""
    if not loss_summary or not alpha_summary:
        return

    labels_list = _get_bin_labels(bin_config)
    if not labels_list:
        labels_list =[{'key': b, 'region': 'Mid', 'lo': 0, 'hi': 1}
                       for b in _get_sorted_bins(loss_summary)]

    active_bins = [lbl['key'] for lbl in labels_list]
    epochs_loss = sorted(set(r['epoch'] for r in loss_summary))
    epochs_alpha = sorted(set(r['epoch'] for r in alpha_summary))
    common_epochs = sorted(set(epochs_loss) & set(epochs_alpha))

    use_epochs = common_epochs if common_epochs else epochs_loss
    has_alpha = bool(common_epochs)
    labels, epoch_to_group = _average_epochs(use_epochs, chunk_size=30)
    n_groups = len(labels)

    alpha_lookup = {}
    if has_alpha:
        for r in alpha_summary:
            alpha_lookup.setdefault((r['epoch'], r['group']), []).append(r['alpha_mean'])

    matrix, counts = np.zeros((len(active_bins), n_groups)), np.zeros((len(active_bins), n_groups))
    for r in loss_summary:
        b = r['confidence_bin']
        if b not in active_bins or r['epoch'] not in epoch_to_group: continue
        g_idx, row, dr = epoch_to_group[r['epoch']], active_bins.index(b), r['drop_rate']

        if has_alpha:
            alpha_avg = float(np.mean(alpha_lookup.get((r['epoch'], b), [1.0])))
            suppression = (1.0 - alpha_avg) * dr
        else:
            suppression = dr

        matrix[row, g_idx] += suppression
        counts[row, g_idx] += 1

    mask = counts > 0
    matrix[mask] /= counts[mask]

    yticklabels = [f"{_short_label(lbl['key'], labels_list)} [{lbl['lo']:.2f}, {lbl['hi']:.2f})" for lbl in labels_list]

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.2), max(5, len(active_bins) * 1.0)))
    annot_fmt = '.4f' if matrix.max() < 0.01 else '.3f'
    sns.heatmap(matrix, ax=ax, xticklabels=labels, yticklabels=yticklabels,
                cmap='YlOrRd', annot=True, fmt=annot_fmt, linewidths=0.5,
                annot_kws={'size': max(7, 11 - len(active_bins) // 3)},
                cbar_kws={'label': '抑制指数 = (1 - α) × 丢弃率'})

    prev_region = None
    for i, lbl in enumerate(labels_list):
        if prev_region is not None and lbl['region'] != prev_region:
            ax.axhline(y=i, color='black', linewidth=2)
        prev_region = lbl['region']

    ax.set_xlabel('轮次范围')
    ax.set_ylabel('置信度分箱')
    ax.set_title('综合抑制热力图')

    plt.tight_layout()
    path = f'{output_dir}/fig4_suppression_heatmap.png'
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_neighborhood_kl_distribution(neighborhood_raw, output_dir):
    """fig5: Density plot of per-node KL divergence from uniform attention"""
    per_region = neighborhood_raw.get('per_region', {})
    epoch_label = neighborhood_raw.get('epoch', '?')

    if not per_region:
        return

    existing = [r for r in NEIGHBORHOOD_REGION_ORDER
                if r in per_region and per_region[r].get('kl')]
    if not existing:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for region in existing:
        kl_vals = np.array(per_region[region]['kl'])
        p99 = np.percentile(kl_vals, 99)
        kl_clipped = np.clip(kl_vals, 0, p99)
        color = NEIGHBORHOOD_COLORS.get(region, '#888888')
        label = f"{region} (median={np.median(kl_vals):.4f})"
        sns.kdeplot(kl_clipped, ax=ax, color=color, label=label,
                    linewidth=2, fill=True, alpha=0.15)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5,
               alpha=0.7, label='均匀基线 (KL=0)')
    ax.set_xlabel('相对于均匀分布的 KL 散度')
    ax.set_ylabel('密度')
    ax.set_title(f'邻域注意力偏离均匀分布\n'
                 f'（轮次 {epoch_label} | 逐节点 KL 散度，仅度 ≥ 2）')
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = f'{output_dir}/fig5_neighborhood_kl_distribution.png'
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_neighborhood_maxratio_boxplot(neighborhood_raw, output_dir):
    """fig6: Box plots of MaxRatio by region"""
    per_region = neighborhood_raw.get('per_region', {})
    epoch_label = neighborhood_raw.get('epoch', '?')

    if not per_region:
        return

    existing = [r for r in NEIGHBORHOOD_REGION_ORDER
                if r in per_region and per_region[r].get('max_ratio')]
    if not existing:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_data, plot_labels, colors = [], [], []
    for region in existing:
        mr_vals = np.array(per_region[region]['max_ratio'])
        p99 = np.percentile(mr_vals, 99)
        clipped = np.clip(mr_vals, 0, p99)
        plot_data.append(clipped)
        n = len(mr_vals)
        median = np.median(mr_vals)
        plot_labels.append(f"{region}\nmed={median:.2f}")
        colors.append(NEIGHBORHOOD_COLORS.get(region, '#888888'))

    bp = ax.boxplot(plot_data, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.7, label='均匀基线 (比值=1)')
    ax.set_xticklabels(plot_labels, fontsize=10)
    ax.set_ylabel('最大注意力权重 / 均匀权重')
    ax.set_title(f'节点邻域注意力集中度\n'
                 f'（轮次 {epoch_label} | MaxRatio = max(α) / (1/deg)）')
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = f'{output_dir}/fig6_neighborhood_maxratio_boxplot.png'
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_attention_deviation_kl(neighborhood_summary, output_dir, smooth=True):
    """fig7a: Mean KL divergence over training epochs"""
    if not neighborhood_summary:
        return

    fig, ax1 = plt.subplots(figsize=(8, 6))

    for region in NEIGHBORHOOD_REGION_ORDER:
        records = [r for r in neighborhood_summary if r['region'] == region]
        if not records:
            continue
        records.sort(key=lambda x: x['epoch'])

        epochs = [r['epoch'] for r in records]
        kl_means = [r['kl_mean'] for r in records]
        n = records[-1]['node_count']

        color = NEIGHBORHOOD_COLORS.get(region, '#888888')
        label = region

        ax1.plot(epochs, kl_means, alpha=0.15, linewidth=0.8, color=color)
        smoothed = _smooth(kl_means) if smooth else kl_means
        ax1.plot(epochs, smoothed, linewidth=1.5, color=color, label=label)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('平均 KL 散度')
    ax1.set_title('邻域注意力偏离')
    ax1.legend(fontsize=9)

    plt.tight_layout()
    path = f'{output_dir}/fig7a_attention_deviation_kl.png'
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_attention_deviation_cv(neighborhood_summary, output_dir, smooth=True):
    """fig7b: CV over training epochs"""
    if not neighborhood_summary:
        return

    fig, ax2 = plt.subplots(figsize=(8, 6))

    for region in NEIGHBORHOOD_REGION_ORDER:
        records = [r for r in neighborhood_summary if r['region'] == region]
        if not records:
            continue
        records.sort(key=lambda x: x['epoch'])

        epochs = [r['epoch'] for r in records]
        cv_means = [r['cv_mean'] for r in records]
        n = records[-1]['node_count']

        color = NEIGHBORHOOD_COLORS.get(region, '#888888')
        label = region

        ax2.plot(epochs, cv_means, alpha=0.15, linewidth=0.8, color=color)
        smoothed = _smooth(cv_means) if smooth else cv_means
        ax2.plot(epochs, smoothed, linewidth=1.5, color=color, label=label)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('注意力权重平均变异系数')
    ax2.set_title('邻域注意力偏离')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    path = f'{output_dir}/fig7b_attention_deviation_cv.png'
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize suppression metrics')
    parser.add_argument('--input', required=True, help='Path to suppression_metrics.json')
    parser.add_argument('--output', default=None, help='Output directory for figures')
    args = parser.parse_args()

    output_dir = args.output or os.path.dirname(args.input)
    os.makedirs(output_dir, exist_ok=True)

    loss_summary, alpha_summary, alpha_raw, bin_config, neighborhood_summary, neighborhood_raw = load_metrics(args.input)
    plot_confidence_histogram(bin_config, loss_summary, output_dir)
    plot_drop_rate_vs_confidence(loss_summary, bin_config, output_dir)
    plot_alpha_distribution(alpha_summary, alpha_raw, bin_config, loss_summary, output_dir)
    plot_training_dynamics_loss(loss_summary, bin_config, output_dir)
    plot_training_dynamics_alpha(alpha_summary, bin_config, output_dir)
    plot_suppression_heatmap(loss_summary, alpha_summary, bin_config, output_dir)
    plot_neighborhood_kl_distribution(neighborhood_raw, output_dir)
    plot_neighborhood_maxratio_boxplot(neighborhood_raw, output_dir)
    plot_attention_deviation_kl(neighborhood_summary, output_dir)
    plot_attention_deviation_cv(neighborhood_summary, output_dir)

if __name__ == '__main__':
    main()
