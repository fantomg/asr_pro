import matplotlib
import matplotlib.pyplot as plt
import mne

# 设置 Matplotlib 后端为 TkAgg
matplotlib.use('TkAgg')


def compare_preprocessing(raw_path, clean_path, tmin=2, tmax=6,
                          baseline=None, figsize=(12, 6)):
    """
    预处理对比可视化函数

    参数：
    raw_path : str
        原始数据路径
    clean_path : str
        预处理后数据路径
    tmin/tmax : float
        时窗范围（秒）
    baseline : tuple or None
        基线校正时段
    figsize : tuple
        图像尺寸
    """
    # 加载数据
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    clean = mne.io.read_raw_fif(clean_path, preload=True)

    # 事件提取
    events_raw, eventid_raw = mne.events_from_annotations(raw)
    events_clean, eventid_clean = mne.events_from_annotations(clean)

    # 过滤有效事件ID
    common_event_ids = {k: v for k, v in eventid_raw.items() if k in eventid_clean}

    # 创建epochs
    epochs_raw = mne.Epochs(raw, events_raw, event_id=common_event_ids,
                            tmin=tmin, tmax=tmax, baseline=baseline,
                            preload=True, event_repeated='merge')
    epochs_clean = mne.Epochs(clean, events_clean, event_id=common_event_ids,
                              tmin=tmin, tmax=tmax, baseline=baseline,
                              preload=True, event_repeated='merge')

    # 生成 evoked 数据
    evoked_raw = epochs_raw.average()
    evoked_clean = epochs_clean.average()

    # 创建对比图
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # 绘制原始数据波形
    evoked_raw.plot(axes=axes[0], show=False, spatial_colors=True,
                    zorder='std', time_unit='s')
    axes[0].set_title("Raw Data", fontweight='bold')

    # **获取原始数据 y 轴范围**
    raw_ylim = axes[0].get_ylim()

    # 绘制预处理后波形
    evoked_clean.plot(axes=axes[1], show=False, spatial_colors=True,
                      zorder='std', time_unit='s')
    axes[1].set_title("Preprocessed Data", fontweight='bold')

    # **应用相同 y 轴范围**
    axes[1].set_ylim(raw_ylim)

    plt.tight_layout()
    plt.show()
    return fig


# 使用示例
if __name__ == '__main__':
    raw_file = './BCICIV_2a_labeled/A08E_raw.fif'
    clean_file = './clean/A08E_pro_eeg.fif'

    # 执行对比
    compare_preprocessing(
        raw_path=raw_file,
        clean_path=clean_file,
        tmin=2,
        tmax=5.9,
        baseline=None,
        figsize=(12, 6)
    )
