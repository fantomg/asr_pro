import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator  # 新增导入

# 设置 Matplotlib 后端为 TkAgg
matplotlib.use('TkAgg')

# 文件夹路径
folder_path = r'C:\Users\38601\Desktop\ica_score'

# 用于存储数据的列表，记录 (数据集名称, 平均ica_score, 标准误, 通道分组)
data_list = []

# 遍历文件夹中的所有 CSV 文件（此部分保持不变）
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, low_memory=False)

        if 'ica_scores' in df.columns:
            ica_scores = df['ica_scores'].dropna()
            first_scores = []
            for score_str in ica_scores:
                parts = score_str.split(',')
                if parts:
                    part = parts[0].strip()
                    try:
                        score = float(part)
                        first_scores.append(score)
                    except ValueError:
                        pass
            if first_scores:
                average_score = np.mean(first_scores)
                if len(first_scores) > 1:
                    error = np.std(first_scores, ddof=1) / np.sqrt(len(first_scores))
                else:
                    error = 0
                label = os.path.splitext(filename)[0]

                if 'Number of Channels' in df.columns:
                    channels_series = df['Number of Channels'].dropna()
                    if not channels_series.empty:
                        try:
                            channels = int(channels_series.iloc[0])
                        except ValueError:
                            channels = None
                    else:
                        channels = None
                else:
                    channels = None

                if channels is not None and channels < 48:
                    channel_group = "Below 48"
                else:
                    channel_group = "48 and above"

                data_list.append((label, average_score, error, channel_group))

# 创建图形部分（主要修改区域）
if data_list:
    df_bar = pd.DataFrame(data_list, columns=['Dataset', 'Average_ica_score', 'Error', 'Channel Group'])

    group_order = ["Below 48", "48 and above"]
    groups_present = df_bar["Channel Group"].unique().tolist()

    if len(groups_present) > 1:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 3), sharex=True)
        for ax, group in zip(axes, group_order):
            group_df = df_bar[df_bar["Channel Group"] == group].sort_values(by='Average_ica_score', ascending=False)
            if not group_df.empty:
                y_pos = np.arange(len(group_df))
                colors = sns.color_palette("Spectral", n_colors=len(group_df))

                # 修改1: 添加误差棒颜色参数
                ax.barh(y_pos, group_df['Average_ica_score'],
                        xerr=group_df['Error'],
                        color=colors,
                        capsize=5,
                        ecolor='#808080')  # 中等灰色

                ax.set_yticks(y_pos)
                ax.set_yticklabels(group_df['Dataset'])
                ax.set_title(f"Channel Group: {group}")
                ax.set_xlabel("Artifact-Free Ratio")
                ax.set_ylabel("Dataset")

                # 修改2: 添加刻度设置
                ax.xaxis.set_major_locator(MultipleLocator(0.1))  # 主刻度间隔0.1
                ax.xaxis.set_minor_locator(MultipleLocator(0.05))  # 次刻度间隔0.05

                # 修改3: 添加网格线设置
                ax.grid(axis='x',
                        which='major',
                        linestyle='--',
                        linewidth=0.5,
                        color='lightgray',
                        alpha=0.7)

                # 设置坐标轴范围
                ax.set_xlim(0, 1.0)
            else:
                ax.set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        group_df = df_bar.sort_values(by='Average_ica_score', ascending=False)
        y_pos = np.arange(len(group_df))
        colors = sns.color_palette("pastel", n_colors=len(group_df))

        # 修改1: 添加误差棒颜色参数
        ax.barh(y_pos, group_df['Average_ica_score'],
                xerr=group_df['Error'],
                color=colors,
                capsize=5,
                ecolor='#808080')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(group_df['Dataset'])
        ax.set_title("Average ICA Scores by Dataset")
        ax.set_xlabel("Artifact-Free Ratio")
        ax.set_ylabel("Dataset")

        # 修改2: 添加刻度设置
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))

        # 修改3: 添加网格线设置
        ax.grid(axis='x',
                which='major',
                linestyle='--',
                linewidth=0.5,
                color='lightgray',
                alpha=0.7)

        ax.set_xlim(0, 1.0)

    plt.tight_layout()
    plt.show()
else:
    print("No valid 'ica_scores' data found in the provided CSV files.")