import os

import mne
import numpy as np
import scipy.io as scio

output_path = r"BCICIV_2a_labeled"  # Output path


def read_eeg_data_from_directory(directory_path):
    """
    读取指定目录下的所有脑电数据文件，并返回包含每个文件路径和其原始数据的列表。

    参数：
    directory_path (str): 包含脑电数据文件的目录路径。

    返回：
    list: 包含每个 EEG 数据文件及其对应原始数据的元组列表。
    """
    # 获取目录下所有文件
    files = os.listdir(directory_path)
    # 筛选出所有的脑电数据文件（根据文件扩展名）
    eeg_files = [f for f in files if f.endswith(('.edf', '.bdf', '.set', '.vhdr', '.cnt', '.gdf'))]

    if not eeg_files:
        raise ValueError(f"在目录 {directory_path} 中未找到支持的脑电数据文件。")

    eeg_data_list = []
    for eeg_file in eeg_files:
        eeg_file_path = os.path.join(directory_path, eeg_file)

        # 根据文件扩展名选择相应的读取函数
        if eeg_file.endswith('.edf'):
            raw = mne.io.read_raw_edf(eeg_file_path)
        elif eeg_file.endswith('.bdf'):
            raw = mne.io.read_raw_bdf(eeg_file_path)
        elif eeg_file.endswith('.set'):
            raw = mne.io.read_raw_eeglab(eeg_file_path)
        elif eeg_file.endswith('.vhdr'):
            raw = mne.io.read_raw_brainvision(eeg_file_path)
        elif eeg_file.endswith('.cnt'):
            raw = mne.io.read_raw_cnt(eeg_file_path)
        elif eeg_file.endswith('.gdf'):
            raw = mne.io.read_raw_gdf(eeg_file_path)
        else:
            raise ValueError(f"不支持的脑电数据文件格式：{eeg_file}")

        eeg_data_list.append((raw, eeg_file))  # 保存原始数据和文件名

    return eeg_data_list


def process_2a(mne_raw, output_dir, eeg_file):
    """
    处理 EEG 数据，添加基于 'pos' 和 'label' 的事件，并保存处理后的数据文件。

    参数：
    mne_raw (mne.io.Raw): 原始的 EEG 数据对象
    output_dir (str): 输出目录
    eeg_file (str): EEG 数据文件名，用于生成输出文件名
    """

    mne_raw.set_channel_types({'EOG-left': 'eog', 'EOG-central': 'eog', 'EOG-right': 'eog'})
    # 重命名通道，按照 10-20 系统重命名
    mne_raw.rename_channels({
        'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz',
        'EEG-3': 'FC2', 'EEG-4': 'FC4', 'EEG-5': 'C5', 'EEG-C3': 'C3',
        'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4',
        'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz',
        'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-14': 'P1', 'EEG-15': 'Pz',
        'EEG-16': 'P2', 'EEG-Pz': 'POz'
    })

    montage = mne.channels.make_standard_montage('standard_1020')
    mne_raw.info.set_montage(montage, on_missing='ignore')

    # Event ID 映射
    event_id = {
        '1023 Rejected trial': 1,
        '1072 Eye movements': 2,
        '276 Idling EEG (eyes open)': 3,
        '277 Idling EEG (eyes closed)': 4,
        '32766 Start of a new run': 5,
        '768 Start of a trial': 6,
        '769 Cue onset left (class 1)': 7,
        '770 Cue onset right (class 2)': 8,
        '771 Cue onset foot (class 3)': 9,
        '772 Cue onset tongue (class 4)': 10
    }

    # 从注释中提取事件
    events, original_event_id = mne.events_from_annotations(mne_raw)

    # 更新事件 ID
    for event_desc, new_id in event_id.items():
        if event_desc in original_event_id:
            events[events[:, 2] == original_event_id[event_desc], 2] = new_id

    # 检查文件名是否以'E'结尾，并加载 .mat 文件替换事件
    file_base, file_ext = os.path.splitext(eeg_file)
    if file_base.endswith('E') and file_ext == '.gdf':
        mat_filepath = f"{file_base}.mat"
        if os.path.exists(mat_filepath):
            mat_data = scio.loadmat(mat_filepath)
            values_from_mat = mat_data[
                                  'classlabel'].flatten() + 6  # Replace 'data' with the correct key in your .mat file

            # 用 .mat 文件中的数据替换事件
            replacement_indices = np.where(events[:, -1] == 7)[0]
            if len(replacement_indices) >= len(values_from_mat):
                events[replacement_indices[:len(values_from_mat)], 2] = values_from_mat
            else:
                print(f"Warning: {mat_filepath} contains fewer values than needed for replacement.")

    # 将更新后的事件转回注释
    event_desc = {value: key for key, value in event_id.items()}
    annotations = mne.annotations_from_events(
        events=events,
        sfreq=mne_raw.info['sfreq'],
        event_desc=event_desc
    )
    mne_raw.set_annotations(annotations)

    # 保存处理后的 EEG 数据
    output_filename = f"{file_base}_raw.fif"
    output_filepath = os.path.join(output_dir, output_filename)
    mne_raw.save(output_filepath, overwrite=True)
    # print(mne_raw.info['ch_names'])
    print(f"处理并保存文件：{output_filename}")


def main():
    # 设置包含脑电数据文件的目录路径
    directory_path = './dataset/s8'

    try:
        # 获取所有 EEG 数据文件
        eeg_data_list = read_eeg_data_from_directory(directory_path)

        # 对每个文件进行处理
        for raw_data, eeg_file in eeg_data_list:
            process_2a(raw_data, output_path, eeg_file)

        print("所有脑电数据处理完成。")

    except Exception as e:
        print(f"发生错误：{e}")


if __name__ == '__main__':
    main()
