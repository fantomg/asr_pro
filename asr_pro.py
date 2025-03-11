import logging
import warnings

import numpy as np
from numpy.linalg import pinv
from scipy import linalg
from scipy.signal import lfilter

from origin_asrpy_utils import (geometric_median, fit_eeg_distribution, yulewalk,
                                yulewalk_filter, ma_filter, block_covariance)
import numpy as np
from scipy.linalg import pinv, eigh
import warnings

from joblib import Parallel, delayed


class ASR():

    def __init__(self, sfreq, cutoff=20, blocksize=100, win_len=0.5,
                 win_overlap=0.66, max_dropout_fraction=0.1,
                 min_clean_fraction=0.25, ab=None, max_bad_chans=0.1,
                 method="euclid"):

        # set attributes
        self.sfreq = sfreq
        self.cutoff = cutoff
        self.blocksize = blocksize
        self.win_len = win_len
        self.win_overlap = win_overlap
        self.max_dropout_fraction = max_dropout_fraction
        self.min_clean_fraction = min_clean_fraction
        self.max_bad_chans = max_bad_chans
        self.method = "euclid"  # NOTE: riemann is not yet available
        self._fitted = False

        # set default yule-walker filter
        if ab is None:
            yw_f = np.array([0, 2, 3, 13, 16, 40,
                             np.minimum(80.0, (self.sfreq / 2.0) - 1.0),
                             self.sfreq / 2.0]) * 2.0 / self.sfreq
            yw_m = np.array([3, 0.75, 0.33, 0.33, 1, 1, 3, 3])
            self.B, self.A = yulewalk(8, yw_f, yw_m)
        else:
            self.A, self.B = ab

        self._reset()

    def _reset(self):
        """Reset state variables."""
        self.M = None
        self.T = None

        # TODO: The following parameters are effectively not used. Still,
        #  they can be set manually via asr.transform(return_states=True)
        self.R = None
        self.carry = None
        self.Zi = None
        self.cov = None
        self._fitted = False

    def fit(self, raw, picks="eeg", start=0, stop=None,
            return_clean_window=False):

        # extract the data
        X = raw.get_data(picks=picks, start=start, stop=stop)

        # Find artifact-free windows first
        clean, sample_mask = clean_windows(
            X,
            sfreq=self.sfreq,
            win_len=self.win_len,
            win_overlap=self.win_overlap,
            max_bad_chans=self.max_bad_chans,
            min_clean_fraction=self.min_clean_fraction,
            max_dropout_fraction=self.max_dropout_fraction)

        # Perform calibration
        self.M, self.T = asr_calibrate(
            clean,
            sfreq=self.sfreq,
            cutoff=self.cutoff,
            blocksize=self.blocksize,
            win_len=self.win_len,
            win_overlap=self.win_overlap,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
            ab=(self.A, self.B),
            method=self.method)

        self._fitted = True

        # return data if required
        if return_clean_window:
            return clean, sample_mask

    def transform(self, raw, picks="eeg", lookahead=0.25, stepsize=32,
                  maxdims=0.66, return_states=False, mem_splits=3):
        """ASR处理（分块版）"""

        X = raw.get_data(picks=picks)
        C, N = X.shape
        P = int(self.sfreq * lookahead)

        # 末尾填充P个零样本
        X_padded = np.pad(X, ((0, 0), (0, P)), mode='constant')

        # 分块处理
        chunk_size = 10000 + P  # 确保每块足够大
        X_processed = np.zeros((C, N))
        self.carry = None  # 重置状态

        for start in range(0, N, chunk_size - P):
            end = min(start + chunk_size, N + P)
            chunk = X_padded[:, start:end]

            processed, states = asr_process_optimized(
                chunk, self.sfreq, self.M, self.T,
                windowlen=self.win_len,
                lookahead=lookahead,
                stepsize=stepsize,
                maxdims=maxdims,
                ab=(self.A, self.B),
                R=self.R,
                Zi=self.Zi,
                cov=self.cov,
                carry=self.carry,
                return_states=True,
                method=self.method,
                mem_splits=mem_splits,
                n_jobs=1
            )

            # 计算有效输出区间
            output_start = start
            output_end = output_start + processed.shape[1] - P
            X_processed[:, output_start:output_end] = processed[:, :-P]

            self.carry = states['carry']

        # 应用处理后的数据
        raw = raw.copy()
        raw.apply_function(lambda x: X_processed, picks=picks, channel_wise=False)
        return raw


def asr_calibrate(X, sfreq, cutoff=20, blocksize=100, win_len=0.5,
                  win_overlap=0.66, max_dropout_fraction=0.1,
                  min_clean_fraction=0.25, ab=None, method='euclid'):
    if method == "riemann":
        warnings.warn("Riemannian ASR is not yet supported. Switching back to"
                      " Euclidean ASR.")
        method == "euclid"

    logging.debug('[ASR] Calibrating...')

    # set number of channels and number of samples
    [nc, ns] = X.shape

    # filter the data
    X, _zf = yulewalk_filter(X, sfreq, ab=ab)

    # window length for calculating thresholds
    N = int(np.round(win_len * sfreq))

    # get block covariances
    U = block_covariance(X, window=blocksize)

    # get geometric median for each block
    # Note: riemann mode is not yet supported, else this could be:
    # Uavg = pyriemann.utils.mean_covariance(U, metric='riemann')
    Uavg = geometric_median(U.reshape((-1, nc * nc)) / blocksize)
    Uavg = Uavg.reshape((nc, nc))

    # get the mixing matrix M
    M = linalg.sqrtm(np.real(Uavg))

    # sort the get the sorted eigenvecotors/eigenvalues
    # riemann is not yet supported, else this could be PGA/nonlinear eigenvs
    D, Vtmp = linalg.eigh(M)
    V = Vtmp[:, np.argsort(D)]  # I think numpy sorts them automatically

    # get the threshold matrix T
    x = np.abs(np.dot(V.T, X))
    offsets = np.int_(np.arange(0, ns - N, np.round(N * (1 - win_overlap))))

    # go through all the channels and fit the EEG distribution
    mu = np.zeros(nc)
    sig = np.zeros(nc)
    for ichan in reversed(range(nc)):
        rms = x[ichan, :] ** 2
        Y = []
        for o in offsets:
            Y.append(np.sqrt(np.sum(rms[o:o + N]) / N))
        mu[ichan], sig[ichan], alpha, beta = fit_eeg_distribution(
            Y, min_clean_fraction, max_dropout_fraction)
    T = np.dot(np.diag(mu + cutoff * sig), V.T)

    logging.debug('[ASR] Calibration done.')
    return M, T


def clean_windows(X, sfreq, max_bad_chans=0.2, zthresholds=[-3.5, 5],
                  win_len=.5, win_overlap=0.66, min_clean_fraction=0.25,
                  max_dropout_fraction=0.1):
    assert 0 < max_bad_chans < 1, "max_bad_chans must be a fraction !"

    # set internal variables
    truncate_quant = [0.0220, 0.6000]
    step_sizes = [0.01, 0.01]
    shape_range = np.arange(1.7, 3.5, 0.15)
    max_bad_chans = np.round(X.shape[0] * max_bad_chans)

    # set data indices
    [nc, ns] = X.shape
    N = int(win_len * sfreq)
    offsets = np.int_(np.round(np.arange(0, ns - N, (N * (1 - win_overlap)))))
    logging.debug('[ASR] Determining channel-wise rejection thresholds')

    wz = np.zeros((nc, len(offsets)))
    for ichan in range(nc):
        # compute root mean squared amplitude
        x = X[ichan, :] ** 2
        Y = np.array([np.sqrt(np.sum(x[o:o + N]) / N) for o in offsets])

        # fit a distribution to the clean EEG part
        mu, sig, alpha, beta = fit_eeg_distribution(
            Y, min_clean_fraction, max_dropout_fraction, truncate_quant,
            step_sizes, shape_range)
        # calculate z scores
        wz[ichan] = (Y - mu) / sig

    # sort z scores into quantiles
    wz[np.isnan(wz)] = np.inf  # Nan to inf
    swz = np.sort(wz, axis=0)

    # determine which windows to remove
    if np.max(zthresholds) > 0:
        mask1 = swz[-(int(max_bad_chans) + 1), :] > np.max(zthresholds)
    if np.min(zthresholds) < 0:
        mask2 = (swz[1 + int(max_bad_chans - 1), :] < np.min(zthresholds))

    # combine the two thresholds
    remove_mask = np.logical_or.reduce((mask1, mask2))
    removed_wins = np.where(remove_mask)

    # reconstruct the samples to remove
    sample_maskidx = []
    for i in range(len(removed_wins[0])):
        if i == 0:
            sample_maskidx = np.arange(
                offsets[removed_wins[0][i]], offsets[removed_wins[0][i]] + N)
        else:
            sample_maskidx = np.vstack((
                sample_maskidx,
                np.arange(offsets[removed_wins[0][i]],
                          offsets[removed_wins[0][i]] + N)
            ))

    # delete the bad chunks from the data
    sample_mask2remove = np.unique(sample_maskidx)
    if sample_mask2remove.size:
        clean = np.delete(X, sample_mask2remove, 1)
        sample_mask = np.ones((1, ns), dtype=bool)
        sample_mask[0, sample_mask2remove] = False
    else:
        sample_mask = np.ones((1, ns), dtype=bool)

    return clean, sample_mask


def asr_process_optimized(data, sfreq, M, T, windowlen=0.5, lookahead=0.25,
                          stepsize=32, maxdims=0.66, ab=None, R=None, Zi=None,
                          cov=None, carry=None, return_states=False,
                          method="euclid", mem_splits=3, n_jobs=1):
    """Optimized ASR implementation with error fixes"""

    # ===== 初始化验证 =====
    if method != "euclid":
        warnings.warn("Only Euclidean ASR supported", RuntimeWarning)

    C, S = data.shape
    maxdims = int(np.round(C * maxdims)) if maxdims < 1 else int(maxdims)

    # ===== 滤波器状态初始化 =====
    if Zi is None:
        _, Zi = yulewalk_filter(data, ab=ab, sfreq=sfreq, zi=np.ones((C, 8)))

    # ===== 数据缓冲处理 =====
    P = int(np.round(lookahead * sfreq))
    if carry is None:
        if data.shape[1] < P:
            pad_width = P - data.shape[1]
            padded_data = np.pad(data, ((0, 0), (0, pad_width)), mode='reflect')
            last_part = padded_data[:, -P:]
        else:
            last_part = data[:, -P:]
        carry = 2 * data[:, [0]] - last_part  # 形状 (C, P)
    data = np.concatenate([carry, data], axis=1)

    # ===== 并行处理函数 =====
    def process_window(Xcov_j, M, T, C, maxdims):
        """处理单个窗口的并行函数"""
        try:
            # 确保输入协方差矩阵对称
            Xcov_j = (Xcov_j + Xcov_j.T) / 2

            # 特征分解
            D, V = eigh(Xcov_j)

            # 确定保留成分
            keep_mask = np.logical_or(
                D < np.sum((T @ V) ** 2, axis=0),
                np.arange(C) < (C - maxdims)
            )
            keep_indices = np.where(keep_mask)[0]

            # 强制至少保留1个成分
            if len(keep_indices) == 0:
                keep_indices = [0]
            k = len(keep_indices)

            # 提取特征向量 (C, k)
            V_kept = V[:, keep_indices]

            # 维度验证
            assert V_kept.shape == (C, k), \
                f"V_kept维度错误: {V_kept.shape} != {(C, k)}"

            # 计算投影矩阵 (k, C)
            A = V_kept.T @ M
            assert A.shape == (k, C), \
                f"A矩阵形状错误: {A.shape} != {(k, C)}"

            # 计算伪逆 (C, k)
            A_pinv = pinv(A, rcond=1e-6)
            assert A_pinv.shape == (C, k), \
                f"A_pinv形状错误: {A_pinv.shape} != {(C, k)}"

            # 分步计算重建矩阵
            R = M @ A_pinv @ V_kept.T @ V.T

            # 最终维度验证
            assert R.shape == (C, C), \
                f"重建矩阵形状错误: {R.shape} != {(C, C)}"

            return R.astype(np.float32), False

        except np.linalg.LinAlgError as e:
            print(f"线性代数错误: {str(e)}")
            return np.eye(C, dtype=np.float32), True

    # ===== 主处理循环 =====
    last_trivial = False
    last_R = np.eye(C, dtype=np.float32)

    for i in range(mem_splits):
        start = i * S // mem_splits
        end = min((i + 1) * S // mem_splits, S)
        i_range = slice(start + P, end + P)

        # 带滤波处理
        X, Zi = yulewalk_filter(data[:, i_range], sfreq=sfreq, ab=ab, zi=Zi)

        # 协方差计算
        X_3d = X[:, np.newaxis, :] * X[np.newaxis, :, :]
        X_flat = X_3d.reshape(C * C, -1)
        Xcov, cov = ma_filter_cumsum(int(windowlen * sfreq), X_flat, cov)

        # 更新点处理
        update_at = np.unique(np.clip(
            np.arange(0, Xcov.shape[1], stepsize),
            0, Xcov.shape[1] - 1
        ))
        Xcov_blocks = Xcov.reshape(C, C, -1)[:, :, update_at]

        # 并行处理
        if Parallel:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_window)(Xcov_blocks[..., j], M, T, C, maxdims)
                for j in range(Xcov_blocks.shape[-1])
            )
        else:
            results = [process_window(Xcov_blocks[..., j], M, T, C, maxdims)
                       for j in range(Xcov_blocks.shape[-1])]

        # 应用重建
        prev_idx = 0
        for j, (R, trivial) in enumerate(results):
            curr_idx = update_at[j] + 1
            if curr_idx <= prev_idx:
                continue

            # 向量化混合操作
            blend = (1 - np.cos(np.linspace(0, np.pi, curr_idx - prev_idx))) / 2
            blend = blend.reshape(1, -1)

            slice_range = slice(prev_idx + P, curr_idx + P)
            data[:, slice_range] = (
                    R @ data[:, slice_range] * blend +
                    last_R @ data[:, slice_range] * (1 - blend)
            ).astype(np.float32)

            last_R, last_trivial = R, trivial
            prev_idx = curr_idx

    # 后处理部分修改为仅裁剪前端P个样本
    P = int(np.round(lookahead * sfreq))
    carry = data[:, -P:] if P > 0 else None  # 保存最后P个样本用于下次处理
    clean_data = data[:, P:]  # 关键修改：仅移除前端P个样本

    if return_states:
        return clean_data, {"M": M, "T": T, "R": R, "Zi": Zi,
                            "cov": cov, "carry": carry}
    else:
        return clean_data


# ===== 缺失函数实现 =====
def yulewalk_filter(X, sfreq, ab=None, zi=None, axis=-1):
    """
    Yule-Walker滤波器实现
    参数：
        X : 输入数据 (C, N)
        ab : (A, B) 滤波器系数元组
        zi : 初始条件
    返回：
        滤波后的数据和最终状态
    """
    if ab is None:
        # 默认滤波器系数（示例值）
        B = np.array([1.0, -2.0, 1.5, -0.8])
        A = np.array([1.0, -1.5, 0.9, -0.2])
    else:
        A, B = ab

    if zi is None:
        zi = np.zeros((X.shape[0], max(len(A), len(B)) - 1))

    # 简化的IIR滤波实现
    X_filtered = np.zeros_like(X)
    for ch in range(X.shape[0]):
        x = X[ch, :]
        y, zf = lfilter(B, A, x, zi=zi[ch])
        X_filtered[ch, :] = y
        zi[ch] = zf

    return X_filtered.astype(np.float32), zi


def ma_filter_cumsum(window_len, data, prev_cumsum=None):
    """优化的移动平均滤波器"""
    if prev_cumsum is None:
        prev_cumsum = np.zeros((data.shape[0], 1), dtype=np.float32)

    cumsum = np.hstack([prev_cumsum, np.cumsum(data, axis=1, dtype=np.float32)])
    ma = (cumsum[:, window_len:] - cumsum[:, :-window_len]) / window_len
    return ma.astype(np.float32), cumsum[:, -window_len + 1:]
