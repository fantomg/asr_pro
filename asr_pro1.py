import logging
import warnings
import numpy as np
from numpy.linalg import pinv
from scipy import linalg
from scipy.linalg import eigh
import pywt

from origin_asrpy_utils import ma_filter, block_covariance, geometric_median, fit_eeg_distribution

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=np.ComplexWarning)


def adaptive_wavelet_filter(X, wavelet='db4', level=None):
    """
    对每个通道进行离散小波变换，自动估计噪声水平并进行软阈值处理，
    实现自适应去噪滤波（保留细节信息）。
    X: 输入数据，形状 (channels, samples)
    返回滤波后的数据（与原数据形状相同）。
    """
    channels, samples = X.shape
    X_filtered = np.zeros_like(X)
    for ch in range(channels):
        coeffs = pywt.wavedec(X[ch, :], wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(samples))
        new_coeffs = [coeffs[0]]
        for detail in coeffs[1:]:
            new_coeffs.append(pywt.threshold(detail, value=uthresh, mode='soft'))
        X_filtered[ch, :] = pywt.waverec(new_coeffs, wavelet)[:samples]
    return X_filtered


def clean_windows(X, sfreq, max_bad_chans=0.2, zthresholds=[-3.5, 5],
                  win_len=0.5, win_overlap=0.66, min_clean_fraction=0.25,
                  max_dropout_fraction=0.1):
    """
    根据窗口内统计量（使用 fit_eeg_distribution 拟合）去除异常窗口，
    返回清洁数据和样本掩码。
    """
    nc, ns = X.shape
    N = int(win_len * sfreq)
    offsets = np.int_(np.round(np.arange(0, ns - N, N * (1 - win_overlap))))
    wz = np.zeros((nc, len(offsets)))
    for ichan in range(nc):
        x = X[ichan, :] ** 2
        Y = [np.sqrt(np.sum(x[o:o + N]) / N) for o in offsets]
        mu, sig, _, _ = fit_eeg_distribution(Y, min_clean_fraction, max_dropout_fraction)
        wz[ichan] = (Y - mu) / sig
    swz = np.sort(wz, axis=0)
    mask1 = swz[-(int(max_bad_chans) + 1), :] > np.max(zthresholds)
    mask2 = swz[1 + int(max_bad_chans - 1), :] < np.min(zthresholds)
    remove_mask = np.logical_or(mask1, mask2)
    removed_wins = np.where(remove_mask)
    sample_maskidx = []
    for i in range(len(removed_wins[0])):
        idxs = np.arange(offsets[removed_wins[0][i]], offsets[removed_wins[0][i]] + N)
        sample_maskidx.extend(idxs)
    sample_mask = np.ones(ns, dtype=bool)
    sample_mask[sample_maskidx] = False
    clean = X[:, sample_mask]
    return clean, sample_mask


def asr_calibrate(X, sfreq, blocksize=100, win_len=0.5, win_overlap=0.66,
                  max_dropout_fraction=0.1, min_clean_fraction=0.25):
    """
    校准阶段：对清洁数据进行滤波、协方差计算、分布拟合，构造门限矩阵 T，
    并计算每个通道的基线 RMS 值用于后续自适应门控。
    """
    nc, ns = X.shape
    X = adaptive_wavelet_filter(X, wavelet='db4', level=None)
    # 计算校准数据的每个通道 RMS 值作为基线
    calib_rms = np.sqrt(np.mean(X ** 2, axis=1))
    N = int(np.round(win_len * sfreq))
    U = block_covariance(X, window=blocksize)
    Uavg = geometric_median(U.reshape((-1, nc * nc)) / blocksize).reshape((nc, nc))
    M = linalg.sqrtm(np.real(Uavg))
    D, Vtmp = linalg.eigh(M)
    V = Vtmp[:, np.argsort(D)]
    x = np.abs(np.dot(V.T, X))
    offsets = np.int_(np.arange(0, ns - N, np.round(N * (1 - win_overlap))))
    mu = np.zeros(nc)
    sig = np.zeros(nc)
    for ichan in range(nc):
        rms = x[ichan, :] ** 2
        Y = [np.sqrt(np.sum(rms[o:o + N]) / N) for o in offsets]
        mu[ichan], sig[ichan], _, _ = fit_eeg_distribution(Y, min_clean_fraction, max_dropout_fraction)
    T = np.dot(np.diag(mu + sig), V.T)
    return M, T, calib_rms


def asr_process_optimized(data, sfreq, M, T, calib_rms, windowlen=0.5, lookahead=0.25,
                          stepsize=32, maxdims=0.66, carry=None, return_states=False,
                          mem_splits=3, auto_gate_factor=0.33):
    """
    处理阶段：采用分段处理和并行窗口计算，对数据应用 ASR 修正，
    加入基于数据分布的自适应门控机制，当通道之间差异较小时跳过修正以节省计算时间。

    参数:
      - calib_rms: 校准阶段得到的每个通道的基线 RMS 值
      - auto_gate_factor: 当通道 RMS 值之间的变异系数低于该阈值时（认为通道间差异较小），跳过修正
    """
    C, S = data.shape
    maxdims = int(C * maxdims) if maxdims < 1 else int(maxdims)
    P = int(lookahead * sfreq)
    data = adaptive_wavelet_filter(data, wavelet='db4', level=None)
    # 如果 carry 不存在，则进行边缘填充
    data = np.pad(data, ((0, 0), (P, 0)), mode='edge') if carry is None else np.concatenate([carry, data], axis=1)

    last_R = np.eye(C, dtype=np.float32)

    def process_window(Xcov_j, M, T, C, maxdims, gate_threshold=0.5):
        try:
            Xcov_j = 0.5 * (Xcov_j + Xcov_j.T)
            # 增强数值稳定性：矩阵指数
            Xcov_j = linalg.expm(0.1 * Xcov_j)
            D, V = eigh(Xcov_j)
            keep_mask = np.logical_or(D < np.sum((T @ V) ** 2, axis=0), np.arange(C) < (C - maxdims))
            keep_indices = np.where(keep_mask)[0]
            V_kept = V[:, keep_indices]
            A = V_kept.T @ M
            A_pinv = pinv(A, rcond=1e-6)
            R = M @ A_pinv @ V_kept.T @ V.T

            # 计算修正矩阵与单位矩阵的差异指标
            correction_magnitude = np.linalg.norm(R - np.eye(C), ord='fro')
            # 如果差异低于预设门限，则返回单位矩阵（不做修正）
            if correction_magnitude < gate_threshold:
                R = np.eye(C, dtype=np.float32)
            return R.astype(np.float32), False
        except Exception as e:
            logging.error(f"Window processing error: {str(e)}")
            return np.eye(C, dtype=np.float32), True

    def should_skip_block(block):
        """
        计算当前块每个通道的 RMS 值，并求其变异系数（标准差/均值）。
        如果变异系数低于 auto_gate_factor，则认为各通道之间差异较小，跳过 ASR 修正。
        """
        block_rms = np.sqrt(np.mean(block ** 2, axis=1))
        cv = np.std(block_rms) / np.mean(block_rms)
        return cv < auto_gate_factor

    for i in range(mem_splits):
        start = i * S // mem_splits
        end = (i + 1) * S // mem_splits
        chunk = data[:, start + P: end + P]

        if should_skip_block(chunk):
            # 如果块内各通道 RMS 变化很小，直接跳过 ASR 修正，使用原始数据
            processed = chunk
            states = {'carry': data[:, -P:]}
        else:
            # 否则执行 ASR 修正
            X = adaptive_wavelet_filter(chunk, wavelet='db4', level=None)
            X_3d = X[:, None] * X[None, :]
            X_flat = X_3d.reshape(C * C, -1)
            Xcov, _ = ma_filter(int(windowlen * sfreq), X_flat, None)
            update_at = np.clip(np.arange(0, Xcov.shape[1], stepsize), 0, Xcov.shape[1] - 1)
            Xcov_blocks = Xcov.reshape(C, C, -1)[..., update_at]
            prev_idx = 0
            for j in range(Xcov_blocks.shape[-1]):
                R, _ = process_window(Xcov_blocks[..., j], M, T, C, maxdims)
                curr_idx = update_at[j] + 1
                blend = (1 - np.cos(np.linspace(0, np.pi, curr_idx - prev_idx))) / 2
                data_slice = data[:, prev_idx + P:curr_idx + P]
                data[:, prev_idx + P:curr_idx + P] = (R @ data_slice * blend.reshape(1, -1) +
                                                      last_R @ data_slice * (1 - blend.reshape(1, -1)))
                last_R = R
                prev_idx = curr_idx
            processed = data[:, P:(end - start) + P]
            states = {'carry': data[:, -P:]}

        # 将处理后的块赋值回原始数据区域
        output_start = start
        output_end = min(output_start + (end - start), S)
        data[:, output_start + P:output_end + P] = processed[:, :output_end - output_start]
        carry = states['carry']

    clean_data = data[:, P:S + P]
    if return_states:
        return clean_data, {'carry': data[:, -P:]}
    return clean_data


class ASR:
    def __init__(self, sfreq, blocksize=100, win_len=0.5, win_overlap=0.66,
                 max_dropout_fraction=0.1, min_clean_fraction=0.25,
                 max_bad_chans=0.1):
        self.sfreq = sfreq
        self.blocksize = blocksize
        self.win_len = win_len
        self.win_overlap = win_overlap
        self.max_dropout_fraction = max_dropout_fraction
        self.min_clean_fraction = min_clean_fraction
        self.max_bad_chans = max_bad_chans
        self._fitted = False
        self._reset()

    def _reset(self):
        self.M = None
        self.T = None
        self.calib_rms = None
        self.carry = None
        self._fitted = False

    def fit(self, raw, picks="eeg", start=0, stop=None, return_clean_window=False):
        X = raw.get_data(picks=picks, start=start, stop=stop)
        clean, sample_mask = clean_windows(X, sfreq=self.sfreq, win_len=self.win_len,
                                           win_overlap=self.win_overlap,
                                           max_bad_chans=self.max_bad_chans,
                                           min_clean_fraction=self.min_clean_fraction,
                                           max_dropout_fraction=self.max_dropout_fraction)
        self.M, self.T, self.calib_rms = asr_calibrate(clean, sfreq=self.sfreq, blocksize=self.blocksize,
                                                       win_len=self.win_len, win_overlap=self.win_overlap,
                                                       max_dropout_fraction=self.max_dropout_fraction,
                                                       min_clean_fraction=self.min_clean_fraction)
        self._fitted = True
        if return_clean_window:
            return clean, sample_mask

    def transform(self, raw, picks="eeg", lookahead=0.25, stepsize=32,
                  maxdims=0.66, return_states=False, mem_splits=3):
        X = raw.get_data(picks=picks)
        C, N = X.shape
        P = int(self.sfreq * lookahead)
        X_padded = np.pad(X, ((0, 0), (0, P)), 'constant')
        X_processed = np.zeros((C, N))
        self.carry = None
        for start in range(0, N, 100):
            end = min(start + 100 + P, N + P)
            chunk = X_padded[:, start:end]
            processed, states = asr_process_optimized(chunk, self.sfreq, self.M, self.T, self.calib_rms,
                                                      windowlen=self.win_len, lookahead=lookahead,
                                                      stepsize=stepsize, maxdims=maxdims,
                                                      carry=self.carry,
                                                      return_states=True, mem_splits=mem_splits)
            output_start = start
            output_end = min(output_start + 100, N)
            X_processed[:, output_start:output_end] = processed[:, :output_end - output_start]
            self.carry = states['carry']
        raw = raw.copy()
        raw.apply_function(lambda x: X_processed, picks=picks, channel_wise=False)
        return raw
