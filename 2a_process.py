import os
import mne
import numpy as np
import scipy.io as scio

# 输出路径
OUTPUT_PATH = r"BCICIV_2a_labeled"

# 文件扩展名与对应的读取函数映射
READER_FUNCS = {
    '.edf': mne.io.read_raw_edf,
    '.bdf': mne.io.read_raw_bdf,
    '.set': mne.io.read_raw_eeglab,
    '.vhdr': mne.io.read_raw_brainvision,
    '.cnt': mne.io.read_raw_cnt,
    '.gdf': mne.io.read_raw_gdf
}

# 通道重命名字典
CHANNEL_RENAME_MAP = {
    'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz',
    'EEG-3': 'FC2', 'EEG-4': 'FC4', 'EEG-5': 'C5', 'EEG-C3': 'C3',
    'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4',
    'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz',
    'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-14': 'P1', 'EEG-15': 'Pz',
    'EEG-16': 'P2', 'EEG-Pz': 'POz'
}

# 事件ID映射
EVENT_ID = {
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


def read_eeg_data_from_directory(directory_path):
    """
    读取指定目录下的所有脑电数据文件，并返回包含每个文件路径和其原始数据的列表。
    """
    eeg_files = [f for f in os.listdir(directory_path)
                 if os.path.splitext(f)[1] in READER_FUNCS]

    if not eeg_files:
        raise ValueError(f"在目录 {directory_path} 中未找到支持的脑电数据文件。")

    eeg_data_list = []
    for eeg_file in eeg_files:
        eeg_file_path = os.path.join(directory_path, eeg_file)
        ext = os.path.splitext(eeg_file)[1]
        try:
            reader_func = READER_FUNCS[ext]
            raw = reader_func(eeg_file_path)
            eeg_data_list.append((raw, eeg_file))
        except Exception as e:
            print(f"读取文件 {eeg_file} 时发生错误：{e}")
    return eeg_data_list


def update_events(mne_raw, eeg_file, events):
    """
    根据 .mat 文件替换特定事件，适用于文件名以 'E' 结尾且扩展名为 .gdf 的情况。
    """
    file_base, file_ext = os.path.splitext(eeg_file)
    if file_base.endswith('E') and file_ext == '.gdf':
        mat_filepath = f"{file_base}.mat"
        if os.path.exists(mat_filepath):
            mat_data = scio.loadmat(mat_filepath)
            values_from_mat = mat_data['classlabel'].flatten() + 6
            replacement_indices = np.where(events[:, -1] == 7)[0]
            if len(replacement_indices) >= len(values_from_mat):
                events[replacement_indices[:len(values_from_mat)], 2] = values_from_mat
            else:
                print(f"Warning: {mat_filepath} contains fewer values than needed for replacement.")
    return events


def process_2a(mne_raw, output_dir, eeg_file):
    """
    处理 EEG 数据：重命名通道、设置 montage、更新事件注释并保存处理后的数据文件。
    """
    # 设置 EOG 通道类型
    mne_raw.set_channel_types({'EOG-left': 'eog', 'EOG-central': 'eog', 'EOG-right': 'eog'})
    # 重命名通道
    mne_raw.rename_channels(CHANNEL_RENAME_MAP)
    # 设置 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    mne_raw.info.set_montage(montage, on_missing='ignore')

    # 提取注释中的事件
    events, original_event_id = mne.events_from_annotations(mne_raw)

    # 更新事件 ID
    for event_desc, new_id in EVENT_ID.items():
        if event_desc in original_event_id:
            events[events[:, 2] == original_event_id[event_desc], 2] = new_id

    # 根据需要更新 events（如 .mat 文件替换）
    events = update_events(mne_raw, eeg_file, events)

    # 将更新后的事件转回注释
    event_desc_rev = {value: key for key, value in EVENT_ID.items()}
    annotations = mne.annotations_from_events(
        events=events,
        sfreq=mne_raw.info['sfreq'],
        event_desc=event_desc_rev
    )
    mne_raw.set_annotations(annotations)

    # 保存处理后的 EEG 数据
    file_base = os.path.splitext(eeg_file)[0]
    output_filename = f"{file_base}_raw.fif"
    output_filepath = os.path.join(output_dir, output_filename)
    mne_raw.save(output_filepath, overwrite=True)
    print(f"处理并保存文件：{output_filename}")


def display_evoked_from_saved_file(file_path):
    """
    从保存的 FIF 文件中加载原始 EEG 数据，提取事件、计算 epoch 并平均，显示 evoked 波形。

    参数:
        file_path (str): 保存的 FIF 文件路径
    """
    try:
        raw = mne.io.read_raw_fif(file_path, preload=True)
    except Exception as e:
        print(f"加载文件 {file_path} 失败：{e}")
        return

    # 提取事件与对应的 event_id
    events, event_id = mne.events_from_annotations(raw)

    # 设置 epoch 的时间窗口参数
    tmin, tmax = 0, 5.9
    epochs = mne.Epochs(
        raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
        baseline=None, preload=True, reject_by_annotation=True,
        event_repeated='merge'  # 或者使用 'merge'
    )

    # 如果存在特定事件，优先展示该事件的 evoked 波形
    target_event = '769 Cue onset left (class 1)'
    if target_event in epochs.event_id:
        evoked = epochs[target_event].average()
    else:
        evoked = epochs.average()

    evoked.plot()


def process_all_files():
    """
    读取目录下所有 EEG 数据文件，处理并保存。
    """
    directory_path = './dataset/s8'
    try:
        eeg_data_list = read_eeg_data_from_directory(directory_path)
        for raw_data, eeg_file in eeg_data_list:
            process_2a(raw_data, OUTPUT_PATH, eeg_file)
        print("所有脑电数据处理完成。")
    except Exception as e:
        print(f"发生错误：{e}")


def main():
    """
    主函数：提供一个简单菜单，用户可以选择数据处理或查看保存文件中的 evoked 波形。
    """

    process_all_files()

    # display_evoked_from_saved_file("BCICIV_2a_labeled/A08E_raw.fif")


if __name__ == '__main__':
    main()
