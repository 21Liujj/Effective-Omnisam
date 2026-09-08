#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成更新的图表
按照论文修改要求生成新的折线图
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置图表样式
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def generate_metrics_line_chart():
    """
    图4：多指标性能变化折线图
    横轴：优化阶段
    纵轴：各项指标数值
    """
    stages = ['Baseline', '+MTC', '+MTC+SMR', '+MTC+SMR+SCM']

    # 数据来自表格
    metrics = {
        'mIoU (%)': [60.22, 60.15, 60.00, 60.84],
        'mAcc (%)': [70.45, 70.38, 70.22, 71.15],
        'PixAcc (%)': [91.23, 91.20, 91.15, 91.52],
        'FWIoU (%)': [83.67, 83.61, 83.52, 84.23],
        'BFScore': [39.59, 39.51, 39.23, 39.86]
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx]
        ax.plot(stages, values, marker='o', linewidth=2.5, markersize=8, color=colors[idx])
        ax.set_xlabel('Optimization Stage', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=15)

        # 标注最高点
        max_idx = np.argmax(values)
        ax.scatter(max_idx, values[max_idx], color='red', s=100, zorder=5)

    # 删除多余的子图
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig('figures/fig_metrics_line_chart.png', dpi=300, bbox_inches='tight')
    print("✓ 生成 fig_metrics_line_chart.png")
    plt.close()


def generate_gpu_resources_with_scm():
    """
    图6：GPU资源利用折线图（包含SCM）
    左Y轴：显存峰值（GB）
    右Y轴：Memory Tokens数量
    """
    methods = ['OmniSAM', '+MTC', '+MTC+SMR', '+MTC+SMR+SCM']
    memory_gb = [32.8, 31.5, 31.3, 31.8]
    tokens = [32768, 8192, 4096, 4096]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左Y轴：显存
    color1 = colors[0]
    ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax1.set_ylabel('GPU Memory (GB)', fontsize=12, fontweight='bold', color=color1)
    line1 = ax1.plot(methods, memory_gb, marker='s', linewidth=2.5, markersize=10,
                     color=color1, label='GPU Memory')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(True, alpha=0.3)

    # 右Y轴：Tokens
    ax2 = ax1.twinx()
    color2 = colors[1]
    ax2.set_ylabel('Memory Tokens', fontsize=12, fontweight='bold', color=color2)
    line2 = ax2.plot(methods, tokens, marker='o', linewidth=2.5, markersize=10,
                     color=color2, label='Memory Tokens')
    ax2.tick_params(axis='y', labelcolor=color2)

    # 图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=11)

    plt.title('GPU Resource Utilization with SCM', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/fig_gpu_resources_with_scm.png', dpi=300, bbox_inches='tight')
    print("✓ 生成 fig_gpu_resources_with_scm.png")
    plt.close()


def generate_token_budget_with_scm():
    """
    图8：Token预算与注意力时间关系折线图（包含SCM）
    横轴：Token预算比例（%）
    左Y轴：注意力时间（秒）
    右Y轴：mIoU（%）
    """
    # 100%-25%为仅SMR的曲线
    budget_pct = [100, 75, 50, 25, 12.5]
    attention_time = [0.85, 0.72, 0.58, 0.45, 0.38]  # 注意力时间随Token预算线性下降

    # mIoU数据（训练后）
    miou_training = [60.22, 60.18, 60.10, 60.03, 60.00]

    # 加入SCM后的mIoU（推理阶段）
    miou_with_scm = [60.51, 60.47, 60.39, 60.32, 60.84]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左Y轴：注意力时间
    color1 = colors[0]
    ax1.set_xlabel('Token Budget (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Attention Time (s)', fontsize=12, fontweight='bold', color=color1)
    line1 = ax1.plot(budget_pct, attention_time, marker='s', linewidth=2.5, markersize=10,
                     color=color1, label='Attention Time')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # 右Y轴：mIoU
    ax2 = ax1.twinx()
    color2 = colors[2]
    color3 = colors[3]
    ax2.set_ylabel('mIoU (%)', fontsize=12, fontweight='bold', color=color2)
    line2 = ax2.plot(budget_pct, miou_training, marker='o', linewidth=2.5, markersize=10,
                     color=color2, label='mIoU (Training)', linestyle='-')
    line3 = ax2.plot(budget_pct, miou_with_scm, marker='^', linewidth=2.5, markersize=10,
                     color=color3, label='mIoU (with SCM)', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)

    # 标注12.5%点（MTC+SMR组合）
    ax1.scatter([12.5], [0.38], color='red', s=150, zorder=5, marker='*')
    ax2.scatter([12.5], [60.84], color='red', s=150, zorder=5, marker='*')
    ax1.annotate('MTC+SMR', xy=(12.5, 0.38), xytext=(20, 0.50),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red', fontweight='bold')

    # 图例
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', fontsize=10)

    plt.title('Token Budget vs Attention Time (with SCM)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/fig_token_budget_with_scm.png', dpi=300, bbox_inches='tight')
    print("✓ 生成 fig_token_budget_with_scm.png")
    plt.close()


if __name__ == '__main__':
    print("开始生成更新的图表...")
    generate_metrics_line_chart()
    generate_gpu_resources_with_scm()
    generate_token_budget_with_scm()
    print("\n所有图表生成完成！")
