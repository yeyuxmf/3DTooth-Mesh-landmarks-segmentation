import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def plot_dth_ablation_data():
    # --- 1. 载入真实消融实验数据 ---
    tau_x = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    tau_map = np.array([78.52, 78.60, 78.62, 78.66, 78.70, 78.65, 78.49, 78.26, 78.16])
    tau_cd = np.array([0.443, 0.442, 0.442, 0.441, 0.440, 0.441, 0.442, 0.443, 0.444])

    # 注: mAR 数据趋势与 mAP 高度一致 (在 0.6-1.2 处达到 63.9+ 的峰值)
    # 为了避免 Y 轴刻度跨度过大导致曲线扁平化，图中主要展示 mAP 与 CD 的博弈关系

    # --- 2. 全局学术绘图设置 (TMI 单栏/通栏标准) ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    color_map = '#005b96'  # 核心指标 mAP (深蓝色)
    color_cd = '#b33939'  # 误差指标 CD (砖红色)
    plaster_gray = '#E5E4E2'  # 石膏灰网格线

    fig, ax1 = plt.subplots(figsize=(5.5, 4.0), dpi=300)

    # --- 3. 绘制左轴 (mAP) ---
    ax1.set_xlabel(r'Distance Threshold $D_{th}$', fontsize=12, fontweight='bold')
    ax1.set_ylabel('mAP (%)', color=color_map, fontsize=12, fontweight='bold')

    line1 = ax1.plot(tau_x, tau_map, marker='o', markersize=7, color=color_map,
                     linewidth=2.5, label='mAP (%)', zorder=3)

    # 设置左轴范围，聚焦在 78.0 到 78.8，放大峰值细节
    ax1.set_ylim(78.0, 78.8)
    ax1.xaxis.set_major_locator(MultipleLocator(0.2))
    ax1.tick_params(axis='y', colors=color_map, labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)

    # 添加石膏灰网格
    ax1.grid(True, linestyle='--', color=plaster_gray, linewidth=1.0, zorder=0)

    # --- 4. 绘制右轴 (Chamfer Distance) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('CD (mm)', color=color_cd, fontsize=12, fontweight='bold')

    line2 = ax2.plot(tau_x, tau_cd, marker='s', markersize=7, color=color_cd,
                     linewidth=2.5, linestyle='-.', label='CD (mm)', zorder=3)

    # 设置右轴范围，聚焦误差波动
    ax2.set_ylim(0.440, 0.445)
    ax2.tick_params(axis='y', colors=color_cd, labelsize=11)

    # --- 5. 图例与高亮 ---
    # 高亮最佳区间 (0.8 到 1.2)
    ax1.axvspan(0.8, 1.2, color=plaster_gray, alpha=0.3, zorder=1)
    ax1.text(1.0, 78.1, 'Optimal\nRegion', horizontalalignment='center',
             fontsize=10, style='italic', color='#555555')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', frameon=True,
               edgecolor=plaster_gray, fontsize=11, ncol=2)

    plt.title('Sensitivity Analysis of Distance Threshold', fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig('D_th_sensitivity.png', format='png', bbox_inches='tight', dpi=300)
    plt.show()

from matplotlib.ticker import MultipleLocator, ScalarFormatter
from matplotlib.ticker import ScalarFormatter


def plot_kreg_ablation():
    # --- 1. 载入消融实验数据 ---
    k_values = np.array([16, 32, 64, 128, 256, 512])
    map_scores = np.array([78.23, 78.59, 78.64, 78.70, 78.65, 78.66])
    tau_cd = np.array([0.445, 0.442, 0.441, 0.440, 0.441, 0.441])  # 修正变量名

    # --- 2. 全局学术绘图设置 (对齐 plot_dth_ablation_data) ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    color_map = '#005b96'  # 核心指标 mAP (深蓝色)
    color_cd = '#b33939'  # 误差指标 CD (砖红色)
    plaster_gray = '#E5E4E2'  # 石膏灰

    fig, ax1 = plt.subplots(figsize=(5.5, 4.0), dpi=300)

    # --- 3. 绘制左轴 (mAP) ---
    ax1.set_xlabel(r'Number of Local Neighbors $k_{reg}$', fontsize=12, fontweight='bold')
    ax1.set_ylabel('mAP (%)', color=color_map, fontsize=12, fontweight='bold')

    # 使用对数轴确保 16-512 均匀分布
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(k_values)
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())

    line1 = ax1.plot(k_values, map_scores, marker='o', markersize=7, color=color_map,
                     linewidth=2.5, label='mAP (%)', zorder=3)

    # 设置左轴范围
    ax1.set_ylim(78.0, 78.8)
    ax1.tick_params(axis='y', colors=color_map, labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.grid(True, linestyle='--', color=plaster_gray, linewidth=1.0, zorder=0)

    # --- 4. 绘制右轴 (Chamfer Distance) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('CD (mm)', color=color_cd, fontsize=12, fontweight='bold')

    # 格式与第一张图的 CD 曲线完全一致
    line2 = ax2.plot(k_values, tau_cd, marker='s', markersize=7, color=color_cd,
                     linewidth=2.5, linestyle='-.', label='CD (mm)', zorder=3)

    # 设置右轴范围
    ax2.set_ylim(0.440, 0.445)
    ax2.tick_params(axis='y', colors=color_cd, labelsize=11)

    # --- 5. 图例与高亮 ---
    # 高亮最优区间 (64 到 256)
    ax1.axvspan(64, 256, color=plaster_gray, alpha=0.3, zorder=1)
    ax1.text(128, 78.1, 'Optimal\nRegion', horizontalalignment='center',
             fontsize=10, style='italic', color='#555555')

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', frameon=True,
               edgecolor=plaster_gray, fontsize=11, ncol=2)

    plt.title('Sensitivity Analysis of Neighbor Count $k_{reg}$', fontsize=13, pad=15)
    plt.tight_layout()

    plt.savefig('K_reg_sensitivity_v2.png', format='png', bbox_inches='tight', dpi=300)
    plt.show()

# 运行


if __name__ == '__main__':
    plot_dth_ablation_data()
    plot_kreg_ablation()