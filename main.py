import asrpy
import mne

from plot.waveform import plot_evoked_joint

# 输入的原始 EEG 数据文件路径
input_file = './BCICIV_2a_labeled/A02E_raw.fif'
output_file = './BCICIV_2a_processed/A02E_asr.fif'


def apply_asr(raw_data):
    """
    使用 asrpy 库进行伪影去除
    :param raw_data: 输入的原始 EEG 数据
    :return: 经过 ASR 处理后的 EEG 数据
    """
    # 创建 ASR 对象，设置采样率和截止频率
    asr = asrpy.ASR(sfreq=raw_data.info['sfreq'], cutoff=20)  # 设置采样率和截止频率

    # 适应数据并应用 ASR 进行伪影去除
    asr.fit(raw_data)  # 适配数据
    raw_cleaned = asr.transform(raw_data)  # 伪影去除

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
    # Get the event times and their corresponding identifiers
    events, events_id = mne.events_from_annotations(raw)

    # Create an MNE Epochs object containing the selected data
    epochs = mne.Epochs(raw, events, events_id, tmin=3, tmax=5.9, proj=True, baseline=None, preload=True,
                        picks=channel_names, event_repeated='drop')
    # 查看原始数据的一些基本信息
    print(f"原始数据的基本信息：{raw.info}")

    # 应用 ASR 进行伪影去除
    raw_cleaned = apply_asr(raw)
    events, events_id = mne.events_from_annotations(raw_cleaned)
    cleaned_avg = mne.Epochs(raw_cleaned, events, events_id, tmin=3, tmax=5.9, proj=True, baseline=None, preload=True,
                             picks=channel_names, event_repeated="drop")
    plot_evoked_joint(
        cleaned_avg,
        epochs,
        times="peaks",
        title=None,
        picks=channel_names,
        exclude=None,
        show=True,
    )
    # 查看处理后的数据的一些基本信息
    print(f"处理后的数据的基本信息：{raw_cleaned.info}")

    # 保存处理后的数据到新的文件
    raw_cleaned.save(output_file, overwrite=True)

    print(f"处理后的数据已保存到：{output_file}")


if __name__ == "__main__":
    main()
