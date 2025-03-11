import os
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.use('TkAgg')

# 文件夹路径
folder_path = r'C:\Users\38601\Desktop\locator'

# 用于存储数据的列表
data_list = []

# 初始化计数器和变量
total_samples = 0
integrity_fail_count = 0
below_80_count = 0
# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        # 读取CSV文件，忽略 DtypeWarning
        df = pd.read_csv(file_path, low_memory=False)
        total_samples += len(df)
        # 计算未通过完整性检查的数量
        integrity_fail_count += len(df[df['Check'] == 'No'])
        # 计算分数低于80的数量
        # 提取Quality Score列并忽略NaN值
        if 'Quality Score' in df.columns:
            quality_scores = df['Quality Score'].dropna()
            below_80_count += len(quality_scores[quality_scores < 80])
            # 只有在有有效数据时才添加到列表
            if not quality_scores.empty:
                # 去掉文件名的扩展名
                label = os.path.splitext(filename)[0]
                # 如果超过200个数据点，随机打乱顺序并取前200个
                if len(quality_scores) > 200:
                    quality_scores = quality_scores.sample(n=200, random_state=42)
                # 将每个文件的每个Quality Score作为一个数据点存储
                for score in quality_scores:
                    data_list.append((label, score))

# 只有在有有效数据时才创建小提琴图
if data_list:
    # 将数据转换为DataFrame
    sns.set_palette('pastel')
    df_violin = pd.DataFrame(data_list, columns=['File', 'Quality Score'])

    # 设置字体和大小
    plt.rc('font', family='Arial', size=12)

    # 创建画布和坐标轴
    fig, ax = plt.subplots(figsize=(7.16, 3.58))
    ax.axvspan(0, 80, facecolor='#FFE6E6')
    ax.axvspan(80, 100, facecolor='#F0FFE9')
    # 添加分界线
    ax.axvline(80, color='blue', linestyle='--', linewidth=1)

    # 将数据按 Quality Score 分成两部分
    df_violin_below_80 = df_violin[df_violin['Quality Score'] < 80]
    df_violin_above_equal_80 = df_violin[df_violin['Quality Score'] >= 80]
    # 计算三个比例
    integrity_fail_ratio = integrity_fail_count / total_samples
    below_80_ratio = below_80_count / total_samples
    remaining_data_ratio = 1 - (integrity_fail_ratio + below_80_ratio)
    # 添加纵向背景颜色，覆盖整个y轴范围
    sns.boxplot(x='Quality Score', y='File', data=df_violin, color='lightgrey', showfliers=False)
    # 绘制小于80分的点
    sns.stripplot(x='Quality Score', y='File', data=df_violin_below_80, jitter=0.3, size=1.3, color='#555555')

    # 绘制大于等于80分的点
    sns.stripplot(x='Quality Score', y='File', data=df_violin_above_equal_80, jitter=0.3, size=1.3, color='black')

    # 计算中位数
    medians = df_violin.groupby('File')['Quality Score'].median().values

    # 绘制中位数线
    for i, median in enumerate(medians):
        plt.plot([median, median], [i - 0.26, i + 0.26], color='red', lw=1.2, zorder=9)
    # Set horizontal axis ticks interval to 5
    ax.xaxis.set_major_locator(MultipleLocator(5))
    # Set horizontal axis minor ticks interval to 1
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    # 显示次刻度
    ax.tick_params(which='both', width=0.5)
    ax.tick_params(which='major', length=5)
    ax.tick_params(axis='y', labelsize=10)
    plt.xlabel('Quality Score')
    plt.ylabel('Dataset')

    # 设置横轴范围
    plt.xlim(40, 100)
    plt.grid(True, linestyle='--', linewidth="0.3")
    # 去除 y 轴的上下留白
    ax.set_ylim(-0.5, len(df_violin['File'].unique()) - 0.5)
    labels = ['Integrity Fail', 'Quality Score Filtered', 'Remaining Data']

    # 在小提琴图内创建一个小坐标轴
    ax_inset = inset_axes(ax, width='40%', height='40%', loc='upper left')

    # 计算百分比
    sizes = [integrity_fail_ratio, below_80_ratio, remaining_data_ratio]
    colors = ['#FF9999', '#FFE6E6', '#F0FFE9']

    # 生成饼图（不使用 autopct）
    wedges, texts = ax_inset.pie(sizes, colors=colors, startangle=140, wedgeprops={'edgecolor': 'black'})

    # 格式化带有百分比的图例标签（保留三位小数）
    labels = [
        f'Integrity Fail :{sizes[0] * 100:.3f}%',
        f'Quality Score Filtered :{sizes[1] * 100:.3f}%',
        f'Remaining Data :{sizes[2] * 100:.3f}%'
    ]

    # 添加图例
    legend = ax_inset.legend(wedges, labels, loc='lower center', bbox_to_anchor=(0.6, -0.8),
                             frameon=True, edgecolor='black')

    # 调整图例文字的字号
    for text in legend.get_texts():
        text.set_fontsize(9)

    # 确保饼图为圆形
    ax_inset.axis('equal')

    # 显示图形
    plt.tight_layout()
    # 显示图形
    plt.show()
else:
    print("No valid 'Quality Score' data found in the provided CSV files.")
