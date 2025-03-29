import time

import awarf

import mne
import matplotlib
from plot.waveform import plot_evoked_joint

matplotlib.use('TkAgg')  # 显式设置 TkAgg 后端

# 输入的原始 EEG 数据文件路径
input_file = './BCICIV_2a_labeled/A04E_raw.fif'
output_file = './clean/A04E_pro_eeg.fif'


def apply_awarf(raw_data):
    """
    使用 asrpy 库进行伪影去除
    :param raw_data: 输入的原始 EEG 数据
    :return: 经过 ASR 处理后的 EEG 数据
    """
    # 创建 ASR 对象，设置采样率（截止频率在 ASR 内部设置）
    start_time = time.time()
    asr = awarf.AWARF(sfreq=raw_data.info['sfreq'])
    asr.fit(raw_data)  # 适应数据
    raw_cleaned = asr.transform(raw_data)  # 进行伪影去除
    end_time = time.time()  # Record end time
    total_time = end_time - start_time  # Calculate total time
    print(f"ASR time：{total_time:.2f}秒")
    return raw_cleaned


def main():
    channel_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz'
    ]

    # 读取原始 EEG 数据
    raw = mne.io.read_raw_fif(input_file, preload=True)

    # 为确保 montage 信息正确，设置标准 10-20 布局
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    # 获取事件信息
    events, events_id = mne.events_from_annotations(raw)

    # 创建 Epochs 对象，选取部分通道进行分析
    epochs = mne.Epochs(raw, events, events_id, tmin=2, tmax=6,
                        proj=True, baseline=None, preload=True,
                        picks=channel_names, event_repeated='drop')

    print(f"原始数据的基本信息：\n{raw.info}")

    # 应用 ASR 进行伪影去除
    raw_cleaned = apply_awarf(raw)

    # 重新获取事件并构建去伪影后的 Epochs 对象
    events_clean, events_id_clean = mne.events_from_annotations(raw_cleaned)
    cleaned_epochs = mne.Epochs(raw_cleaned, events_clean, events_id_clean,
                                tmin=2, tmax=6, proj=True, baseline=None,
                                preload=True, picks=channel_names, event_repeated="drop")

    # 绘制去伪影前后数据对比
    plot_evoked_joint(
        cleaned_epochs,
        epochs,
        times="peaks",
        title=None,
        picks=channel_names,
        exclude=None,
        show=True,
    )

    print(f"处理后的数据的基本信息：\n{raw_cleaned.info}")

    # 保存处理后的数据
    raw_cleaned.save(output_file, overwrite=True)
    print(f"处理后的数据已保存到：{output_file}")


if __name__ == "__main__":
    main()
