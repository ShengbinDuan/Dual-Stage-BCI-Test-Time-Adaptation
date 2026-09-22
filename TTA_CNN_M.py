import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function
from typing import Optional, Sequence, Any, Tuple, List, Dict, Callable
from scipy.io import loadmat
from scipy.linalg import fractional_matrix_power
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import random
from tqdm import tqdm
import logging
from math import log2
import os
import pickle
import csv
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ----------------- 常量定义 -----------------
SAMPLING_RATE     = 250
WINDOW_LENGTH     = 250
SLIDING_STEP      = 250
TEST_SLIDING_STEP = 250
DATA_PATH         = '../../database/'
TIME_WINDOWS      = 1

# ----------------- 工具函数 -----------------
def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    return torch.sum(entropy, dim=1)

def band_filter(x):
    return x

def EA_alignment(X, return_ref=False):
    cov_matrices = np.array([np.cov(trial) for trial in X])
    mean_cov = np.mean(cov_matrices, axis=0)
    transform = fractional_matrix_power(mean_cov, -0.5)
    X_aligned = np.array([transform @ trial for trial in X])
    if return_ref:
        return X_aligned, mean_cov
    else:
        return X_aligned

def process_data(rawdata, mode, window_length, step, apply_EA=False, perform_fft=True, return_ref=False):
    channels = [60,61,62,54,55,56,53,47,57]
    slice_start = round(5/0.2)
    slice_end   = round(55/0.2) + 1
    raw_segments, label_list = [], []
    data_length = 1250
    for block in range(6):
        for trial in range(40):
            filtered = [band_filter(rawdata[ch,160:1410,trial,block]) for ch in channels]
            for start_idx in range(0, data_length-window_length+1, step):
                segment = np.stack([sig[start_idx:start_idx+window_length] for sig in filtered])
                raw_segments.append(segment)
                label_list.append(trial)
    raw_segments = np.array(raw_segments)
    if apply_EA:
        if return_ref:
            aligned_segments, ref = EA_alignment(raw_segments, return_ref=True)
        else:
            aligned_segments = EA_alignment(raw_segments)
    else:
        aligned_segments = raw_segments
        if return_ref:
            ref = None

    if not perform_fft:
        return (aligned_segments, np.array(label_list), ref) if return_ref else (aligned_segments, np.array(label_list))
    else:
        data_list = []
        for segment in aligned_segments:
            trial_channels = []
            for sig in segment:
                fft_res = np.fft.rfft(sig, n=1250) / window_length
                if mode == "abs":
                    fft_res = np.abs(fft_res)[slice_start:slice_end]
                else:
                    fft_res = np.concatenate((np.real(fft_res)[slice_start:slice_end],
                                              np.imag(fft_res)[slice_start:slice_end]))
                trial_channels.append(fft_res)
            data_list.append(np.array(trial_channels))
        data_list = np.expand_dims(np.array(data_list), axis=-1)
        return (data_list, np.array(label_list), ref) if return_ref else (data_list, np.array(label_list))

def get_cache_filename(mode, subject, time_length):
    os.makedirs("cache", exist_ok=True)
    return os.path.join("cache", f"cache_{mode}{subject}{time_length}.pkl")

def data_batch_FFT_one_tester(mode, test_trainer, time_length, apply_EA=False, return_raw_test=False):
    cache_file = get_cache_filename(mode, test_trainer, time_length)
    if os.path.exists(cache_file):
        return pickle.load(open(cache_file, "rb"))
    subjects = [f"S{i}" for i in range(1,36)]
    window_length = int(SAMPLING_RATE * time_length)
    train_data_list, train_labels_list, train_refs = [], [], []
    test_index = subjects.index(test_trainer)
    train_subjects = subjects[:test_index] + subjects[test_index+1:]
    for subj in train_subjects:
        rawdata = loadmat(DATA_PATH + subj + '.mat')['data']
        d, l, ref = process_data(rawdata, mode, window_length, SLIDING_STEP, apply_EA=True, perform_fft=True, return_ref=True)
        train_data_list.append(d)
        train_labels_list.append(l)
        train_refs.append(ref)
    train_data   = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    train_ref    = np.mean(np.array(train_refs), axis=0)
    rawdata = loadmat(DATA_PATH + test_trainer + '.mat')['data']
    if return_raw_test:
        test_data, test_labels = process_data(rawdata, mode, window_length, TEST_SLIDING_STEP, apply_EA=False, perform_fft=False)
    else:
        test_data, test_labels = process_data(rawdata, mode, window_length, TEST_SLIDING_STEP, apply_EA=True, perform_fft=True)
    data_tuple = (train_data, train_labels, test_data, test_labels, train_ref)
    pickle.dump(data_tuple, open(cache_file, "wb"))
    return data_tuple

def data_batch_FFT_one_tester_abs_UI(test_trainer, time_length):
    return data_batch_FFT_one_tester("abs", test_trainer, time_length, apply_EA=True, return_raw_test=True)

def data_batch_FFT_one_tester_complex_UI(test_trainer, time_length):
    return data_batch_FFT_one_tester("comp", test_trainer, time_length, apply_EA=True, return_raw_test=True)

def EA_online(x, R, sample_num, weight=500):
    cov = np.cov(x)
    new_sample_count = sample_num + weight
    return (R * sample_num + weight * cov) / new_sample_count

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------- 模型定义 --------------
class RobustBN(nn.Module):
    @staticmethod
    def find_bns(parent, alpha):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            if isinstance(child, nn.BatchNorm2d):
                replace_mods.append((parent, name, RobustBN(child, alpha)))
            else:
                replace_mods.extend(RobustBN.find_bns(child, alpha))
        return replace_mods

    @staticmethod
    def adapt_model(model, alpha):
        for parent, name, module in RobustBN.find_bns(model, alpha):
            setattr(parent, name, module)
        return model

    def __init__(self, bn_layer: nn.BatchNorm2d, momentum):
        super().__init__()
        self.momentum = momentum
        if bn_layer.track_running_stats:
            self.register_buffer("source_mean", bn_layer.running_mean.clone())
            self.register_buffer("source_var", bn_layer.running_var.clone())
            self.source_num = bn_layer.num_batches_tracked
        self.weight = bn_layer.weight.clone()
        self.bias   = bn_layer.bias.clone()
        self.eps    = bn_layer.eps

    def forward(self, x, trial_lengths=None):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0,2,3], unbiased=False)
            mean = (1-self.momentum)*self.source_mean + self.momentum*b_mean
            var  = (1-self.momentum)*self.source_var  + self.momentum*b_var
            self.source_mean.copy_(mean.detach())
            self.source_var.copy_(var.detach())
        else:
            mean, var = self.source_mean, self.source_var
        mean = mean.view(1,-1,1,1); var = var.view(1,-1,1,1)
        x = (x-mean)/torch.sqrt(var+self.eps)
        return x * self.weight.view(1,-1,1,1) + self.bias.view(1,-1,1,1)

class CNNModel(nn.Module):
    def __init__(self, input_width, decay, num_classes=40):
        super().__init__()
        self.conv1    = nn.Conv2d(1,18,(9,1),bias=False)
        self.bn1      = nn.BatchNorm2d(18)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2    = nn.Conv2d(18,18,(1,15),bias=False)
        self.bn2      = nn.BatchNorm2d(18)
        self.dropout2 = nn.Dropout(0.25)
        self.flat_features = 18*(input_width-14)
        self.fc       = nn.Linear(self.flat_features, num_classes, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.01)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.dropout2(x)
        x = x.flatten(1)
        return self.fc(x)

def itr(n, p, t):
    if p < 1.0/n:
        return 0.0
    elif p == 1:
        return log2(n)*60.0/t
    else:
        return (log2(n) + p*log2(p) + (1-p)*log2((1-p)/(n-1))) * 60.0/t
def main():

    set_seed(2025)
    MODE = "abs"  # "abs" or "comp"
    if MODE == "abs":
        input_width = 251
        data_func   = data_batch_FFT_one_tester_abs_UI
    else:
        input_width = 502
        data_func   = data_batch_FFT_one_tester_complex_UI

    save_dir = f"CNN_{MODE}"
    os.makedirs(save_dir, exist_ok=True)

    log_dir = f"CNN_result_{MODE}"
    os.makedirs(log_dir, exist_ok=True)

    sto_result_dir = "STO_result"
    os.makedirs(sto_result_dir, exist_ok=True)

    # 存放每位被试的 TTA 曲线（txt）
    curve_dir = os.path.join(sto_result_dir, "TTA_curves")
    os.makedirs(curve_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 一次 TTA 配置（前半段）；后半段为固定验证集
    BATCH_LIST = [1]
    STRIDE            = 3
    t_val             = 6.0
    epsilon           = 1e-2
    online_lr         = 3e-5

    subjects = [f"S{i}" for i in range(1,36)]
    FIXED_VAL_ACC = 0.69   # 构造 p_dist 的固定验证 ACC（不从数据估计）

    # 5–55s FFT slice indices（与原实现保持一致）
    slice_start = round(5/0.2)
    slice_end   = round(55/0.2) + 1

    lam_list = [1.1]

    # ----------------- 小工具：在固定验证集上评估 ACC（不更新任何内容） -----------------
    def evaluate_fixed_val(model, test_datas, test_labels, start_idx, end_idx, sqrtR_current):
        """
        使用当前 sqrtR_current 对后半段 [start_idx, end_idx) 的样本逐一对齐 + 特征抽取 + 前向，
        返回 ACC（仅用于“固定验证集”的评估；不做 BN/参数/R 更新）。
        """
        model.eval()
        correct = 0
        total = max(end_idx - start_idx, 0)
        if total == 0:
            return 0.0
        with torch.no_grad():
            for i in range(start_idx, end_idx):
                sample = test_datas[i]
                aligned_sample = sqrtR_current.dot(sample)

                # 特征
                if MODE == "abs":
                    feats = [
                        np.abs(np.fft.rfft(sig, n=1250) / WINDOW_LENGTH)[slice_start:slice_end]
                        for sig in aligned_sample
                    ]
                else:
                    feats = []
                    for sig in aligned_sample:
                        fft_res = np.fft.rfft(sig, n=1250) / WINDOW_LENGTH
                        real_part = np.real(fft_res)[slice_start:slice_end]
                        imag_part = np.imag(fft_res)[slice_start:slice_end]
                        feats.append(np.concatenate((real_part, imag_part)))

                xt = torch.tensor(np.stack(feats), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                out = model(xt)
                pred = int(torch.argmax(out, dim=1).item())
                if pred == int(test_labels[i]):
                    correct += 1
        return correct / total

    # ----------------- 主循环 -----------------
    for batch in BATCH_LIST:
        BN_BATCH = batch
        TEST_UPDATE_BATCH = batch

        lam_acc_summary = []
        batch_acc_summary = []

        for lam_use in lam_list:
            print(f"\n=== TTA with BN_BATCH = {BN_BATCH}, TEST_UPDATE_BATCH = {TEST_UPDATE_BATCH}, lam_use = {lam_use} ===")
            logging.info("Grid search lam_use: %.2f", lam_use)

            per_subject_metrics = []
            subject_accuracies = []

            for subject in subjects:
                print(f"--- Subject {subject} ---")
                logging.info(
                    "Subject: %s, lam_use: %.2f, TEST_UPDATE_BATCH: %d, BN_TEST_UPDATE_BATCH: %d",
                    subject, lam_use, TEST_UPDATE_BATCH, BN_BATCH
                )

                # ----- 载入数据（训练参考与被试原始测试序列）-----
                train_datas, train_labels, test_datas, test_labels, train_ref = data_func(subject, time_length=1)
                print(" Train data shape:", train_datas.shape)
                print(" Test raw data shape:", test_datas.shape)

                # ----- 载入预训练模型 -----
                model = CNNModel(input_width=input_width, decay=1e-4, num_classes=40).to(device)
                ckpt_path = os.path.join(save_dir, f"{MODE}_{online_lr}_600_0.5_{subject}.pth")
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                print(" Loaded weights from:", ckpt_path)

                # ===== 划分：前半段（TTA），后半段（固定验证） =====
                num_total = len(test_datas)
                split_idx = num_total // 2 if num_total > 1 else num_total
                val_start, val_end = split_idx, num_total  # 后半段区间

                # ===== 阶段 A：前半段做 TTA；每次“参数更新(step)”后，在固定验证集上评估并打印 =====
                R = train_ref.copy()
                aligned_buffer = []
                optimizer_tta = optim.SGD(model.parameters(), lr=online_lr, momentum=0.9, weight_decay=1e-4)
                model.eval()

                tta_curve = []  # 存放“每次参数更新后的固定验证集 ACC”
                tta_step  = 0

                for i in range(split_idx):
                    sample = test_datas[i]

                    # 在线更新对齐参考（仅 TTA 阶段）
                    R = EA_online(sample, R, i, weight=5000)
                    sqrtR_current = fractional_matrix_power(R, -0.5)
                    aligned_sample = sqrtR_current.dot(sample)

                    # 抽取特征
                    if MODE == "abs":
                        feats = [
                            np.abs(np.fft.rfft(sig, n=1250) / WINDOW_LENGTH)[slice_start:slice_end]
                            for sig in aligned_sample
                        ]
                    else:
                        feats = []
                        for sig in aligned_sample:
                            fft_res = np.fft.rfft(sig, n=1250) / WINDOW_LENGTH
                            real_part = np.real(fft_res)[slice_start:slice_end]
                            imag_part = np.imag(fft_res)[slice_start:slice_end]
                            feats.append(np.concatenate((real_part, imag_part)))
                    aligned_buffer.append(np.stack(feats))

                    # BN 统计更新（仅 TTA 阶段）
                    if len(aligned_buffer) >= BN_BATCH and (i+1) % STRIDE == 0:
                        model.train()
                        bn_batch_np = np.stack(aligned_buffer[-BN_BATCH:])
                        bn_batch_t  = torch.tensor(bn_batch_np, dtype=torch.float32).unsqueeze(1).to(device)
                        with torch.no_grad():
                            _ = model(bn_batch_t)
                        model.eval()

                    # 参数更新（仅 TTA 阶段）
                    if len(aligned_buffer) >= TEST_UPDATE_BATCH and (i+1) % STRIDE == 0:
                        model.train()
                        batch_np = np.stack(aligned_buffer[-TEST_UPDATE_BATCH:])
                        batch_t  = torch.tensor(batch_np, dtype=torch.float32).unsqueeze(1).to(device)

                        out_b  = model(batch_t)
                        soft_b = F.softmax(out_b / t_val, dim=1)
                        preds  = soft_b.argmax(dim=1)

                        B, C = soft_b.shape
                        p_dist = torch.full((B, C), fill_value=(1-FIXED_VAL_ACC)/(C-1), device=soft_b.device)
                        p_dist[torch.arange(B), preds] = FIXED_VAL_ACC

                        ENT_loss = torch.mean(Entropy(soft_b))
                        CE_loss  = -torch.mean(torch.sum(p_dist * torch.log(soft_b + epsilon), dim=1))

                        optimizer_tta.zero_grad()
                        total_loss = ENT_loss + lam_use * (CE_loss - ENT_loss)
                        total_loss.backward()
                        optimizer_tta.step()
                        model.eval()

                        # ******** 在固定验证集上评估并打印（曲线点）********
                        tta_step += 1
                        # 注意：验证集对齐使用“当前”的 sqrtR_current（不在验证集上更新 R/BN/参数）
                        acc_now = evaluate_fixed_val(model, test_datas, test_labels, val_start, val_end, sqrtR_current)
                        tta_curve.append(acc_now)
                        print(f"[{subject}] TTA step {tta_step:03d} -> Fixed-VAL ACC: {acc_now:.4f}")

                # ===== 阶段 B：最终一次固定验证（可选；使用最后的 R）=====
                if val_end - val_start > 0:
                    sqrtR_val = fractional_matrix_power(R.copy(), -0.5)  # 使用最终 R
                    final_acc = evaluate_fixed_val(model, test_datas, test_labels, val_start, val_end, sqrtR_val)
                else:
                    final_acc = 0.0
                print(f"[{subject}] Final Fixed-VAL ACC (after TTA): {final_acc:.4f}")

                # ===== 保存该被试的 TTA 曲线到 txt（s{idx} = np.array([...])）=====
                sid = int(subject[1:])
                varname = f"s{sid}"
                # 构造格式化字符串（每行最多 10 个，四位小数）
                lines = [f"{varname} = np.array(["]
                if len(tta_curve) == 0:
                    lines.append("])")
                else:
                    for k in range(0, len(tta_curve), 10):
                        chunk = ", ".join(f"{v:.4f}" for v in tta_curve[k:k+10])
                        if k + 10 < len(tta_curve):
                            lines.append(f"    {chunk},")
                        else:
                            lines.append(f"    {chunk}")
                    lines.append("])")
                text_out = "\n".join(lines) + "\n"

                curve_path = os.path.join(curve_dir, f"{subject}_abs_tta_curve.txt")
                with open(curve_path, "w") as f:
                    f.write(text_out)
                print(f"[{subject}] Saved TTA curve to: {curve_path}")
                # 也在控制台打印一份，方便直接复制
                print(text_out)

                # 收集用于总体汇总（这里使用“最终一次固定验证 ACC”）
                per_subject_metrics.append({
                    "subject": subject,
                    "accuracy": final_acc,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "auc": 0.0,
                })
                subject_accuracies.append(final_acc)

            # ----- 汇总每个 lam_use 的平均 ACC（基于最终固定验证）-----
            average_acc = float(np.mean(subject_accuracies)) if len(subject_accuracies) > 0 else 0.0
            print(f"\nlam_use = {lam_use}, 平均（固定验证集） ACC: {average_acc:.4f}")
            logging.info("lam_use: %.2f, average ACC (fixed val): %.4f", lam_use, average_acc)

            lam_acc_summary.append((lam_use, average_acc))
            batch_acc_summary.append((TEST_UPDATE_BATCH, average_acc))

            # ----- 保存每个被试的最终固定验证指标（这里只保存 ACC 字段，其它置 0）-----
            summary_filename = f"{BN_BATCH}_{lam_use:.1f}_{average_acc:.4f}_CNN_{MODE}_FixedVAL.csv"
            summary_path = os.path.join(sto_result_dir, summary_filename)
            with open(summary_path, "w", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow(["subject", "accuracy", "precision", "recall", "f1", "auc", "roc"])
                for metrics in per_subject_metrics:
                    writer.writerow([
                        metrics["subject"],
                        f"{metrics['accuracy']:.4f}",
                        f"{metrics['precision']:.4f}",
                        f"{metrics['recall']:.4f}",
                        f"{metrics['f1']:.4f}",
                        f"{metrics['auc']:.4f}",
                        f"{metrics['auc']:.4f}",
                    ])
            print("Saved per-subject fixed validation metrics to:", summary_path)

        # ----- 保存 lam_use vs 平均 ACC（固定验证集）-----
        lam_acc_filename = f"{BN_BATCH}_CNN_{MODE}_lam_acc_fixedval.csv"
        lam_acc_path = os.path.join(sto_result_dir, lam_acc_filename)
        with open(lam_acc_path, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["lam_use", "average_accuracy_fixed_val"])
            for lam_val, avg in lam_acc_summary:
                writer.writerow([f"{lam_val:.1f}", f"{avg:.4f}"])
        print("\nSaved lam_use vs average ACC (fixed val) to:", lam_acc_path)

        # ----- 保存 batch_size vs 平均 ACC（固定验证集）-----
        batch_summary_path = os.path.join(sto_result_dir, f"{BN_BATCH}_batch_size_average_acc_fixedval.csv")
        with open(batch_summary_path, "w", newline="") as bf:
            writer = csv.writer(bf)
            writer.writerow(["test_update_batch", "average_accuracy_fixed_val"])
            for batch_size, avg_acc in batch_acc_summary:
                writer.writerow([batch_size, f"{avg_acc:.4f}"])
        print("Saved test batch size vs average ACC (fixed val) to:", batch_summary_path)
if __name__ == '__main__':
    main()