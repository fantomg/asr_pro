import mne
import matplotlib

def compute_quality_score_using_mne(raw, method="ica", plot=False):
    """
    Use MNE's built-in artifact detection methods to compute a data quality score.

    Parameters:
    raw: mne.io.Raw
        The EEG raw data.
    method: str
        The method to use for artifact detection. Options: "ica", "maxwell".
    plot: bool
        Whether to plot artifact scores and diagnostics.

    Returns:
    results: dict
        Dictionary containing the quality score, artifact ratio, and individual artifact counts.
    """
    raw_copy = raw.copy().load_data()

    match method:
        case "ica":
            # Set ICA components to the number of EEG channels
            n_components = len([ch for ch in raw_copy.info['chs'] if ch['kind'] == 2])  # 2 for EEG channels

            # Use ICA to detect components with artifacts
            ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, max_iter="auto")
            ica.fit(raw_copy)

            # Initialize scores
            eog_scores = []
            emg_scores = []
            ecg_scores = []

            # Check if EOG channels are present in the dataset
            eog_inds = []
            eog_channels = [ch for ch in raw_copy.info['ch_names'] if 'eog' in ch.lower()]
            if eog_channels:
                eog_inds, eog_scores = ica.find_bads_eog(raw_copy)

            # Check if EMG channels are present in the dataset
            emg_inds = []
            try:
                emg_inds, emg_scores = ica.find_bads_muscle(raw_copy)
            except RuntimeError as e:
                print(f"Warning: {e}. EMG detection skipped.")

            # Check if ECG channels are present in the dataset
            ecg_inds = []
            ecg_channels = [ch for ch in raw_copy.info['ch_names'] if 'ecg' in ch.lower()]
            if ecg_channels:
                try:
                    ecg_inds, ecg_scores = ica.find_bads_ecg(raw_copy)
                except RuntimeError as e:
                    print(f"Warning: {e}. ECG detection skipped.")

            # Combine all detected artifact indices
            artifact_inds = list(set(eog_inds + emg_inds + ecg_inds))
            artifact_ratio = len(artifact_inds) / n_components

            results = {
                "artifact_ratio": artifact_ratio,
                "quality_score": max(0, 1 - artifact_ratio),
                "eog_count": len(eog_inds),
                "emg_count": len(emg_inds),
                "ecg_count": len(ecg_inds)
            }

            # Plot artifact scores if enabled
            if plot:
                if len(eog_scores) > 0:  # Check if EOG scores exist
                    ica.plot_scores(eog_scores, title="EOG Artifact Scores")
                    print("eog_inds: ", eog_inds)
                    ica.plot_properties(raw, picks=eog_inds)
                    ica.plot_overlay(raw, exclude=[0], picks="eeg")

                if len(emg_scores) > 0:  # Check if EMG scores exist
                    ica.plot_scores(emg_scores, title="EMG Artifact Scores")
                    print("emg_inds: ", emg_inds)
                    ica.plot_properties(raw, picks=emg_inds)
                    ica.plot_overlay(raw, exclude=[3], picks="eeg")

                if len(ecg_scores) > 0:  # Check if ECG scores exist
                    ica.plot_scores(ecg_scores, title="ECG Artifact Scores")
                    ica.plot_overlay(raw, exclude=[1], picks="eeg")

        case "maxwell":
            # Detect bad channels using Maxwell filtering
            bads = mne.preprocessing.find_bad_channels_maxwell(raw_copy, verbose=False)
            artifact_ratio = len(bads) / len(raw.ch_names)
            results = {
                "artifact_ratio": artifact_ratio,
                "quality_score": max(0, 1 - artifact_ratio),
                "eog_count": 0,
                "emg_count": 0,
                "ecg_count": 0
            }

        case _:
            raise ValueError(f"Unknown method: {method}")

    return results


def main():
    matplotlib.use('TkAgg')
    # File path to the dataset
    file_path = "/gpfs/work/int/chengxuanqin21/science_works/standard_raweeg/bcic_iv_2a/A01E_raw.fif"
    try:
        raw = mne.io.read_raw(file_path, preload=True)
        # raw.rename_channels(
        #     {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
        #      'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6',
        #      'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4',
        #      'EEG-14': 'P1', 'EEG-15': 'Pz', 'EEG-16': 'P2', 'EEG-Pz': 'POz'})
        # raw.set_channel_types({'EOG-left': 'eog', 'EOG-central': 'eog', 'EOG-right': 'eog'})

        # Add standard montage to fix missing electrode locations
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage)

        print(f"Loaded dataset from {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    # Choose the method for artifact detection
    method = "ica"  # Options: "maxwell", "ica"

    # Compute artifact ratios and quality score using MNE
    results = compute_quality_score_using_mne(raw, method=method)

    # Output results
    print(f"Artifact Detection Method: {method}")
    print(f"Artifact Ratio: {results['artifact_ratio']:.2%}")
    print(f"Data Quality Score: {results['quality_score']:.2f}")
    print(f"EOG Artifact Count: {results['eog_count']}")
    print(f"EMG Artifact Count: {results['emg_count']}")
    print(f"ECG Artifact Count: {results['ecg_count']}")


if __name__ == "__main__":
    main()
