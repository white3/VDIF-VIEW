import numpy as np
import matplotlib.pyplot as plt
from baseband import vdif
from astropy.time import Time
from astropy import units as u
import os

# 配置参数
sample_rate = 4_000_000  # 4 MHz
duration = 2  # 秒
nchan = 2
nbit = 2
complex_data = False
frame_length = 8032  # 字节
samples_per_frame = 1024
total_samples = duration * sample_rate

# 生成模拟信号（正弦波 + 噪声）
t = np.arange(total_samples) / sample_rate
signal_ch0 = 0.5 * np.sin(2 * np.pi * 1e6 * t) + 0.1 * np.random.randn(total_samples)
signal_ch1 = 0.5 * np.cos(2 * np.pi * 1e6 * t) + 0.1 * np.random.randn(total_samples)
original_data = np.stack([signal_ch0, signal_ch1])  # 形状为 (2, total_samples)

# 写入 VDIF 文件
out_path = "sim_real.vdif"
if os.path.exists(out_path):
    os.remove(out_path)

with vdif.open(out_path, 'ws',
               sample_rate=sample_rate*u.Hz,
               nchan=nchan,
               bps=nbit,
               complex_data=complex_data,
               frame_length=frame_length,
               time=Time.now()) as fh:
    for i in range(0, total_samples, samples_per_frame):
        frame = original_data[:, i:i + samples_per_frame]
        if frame.shape[1] < samples_per_frame:
            break  # 跳过最后不足一帧的数据
        fh.write(frame.T.astype(np.float32))  # 转置为 (samples_per_frame, nchan)

# 读取 VDIF 数据
with vdif.open(out_path, 'rs') as fh:
    decoded_data = fh.read(samples_per_frame)  # 读取前 1024 个样本，形状为 (1024, 2)

# 转置原始数据以匹配读取的数据形状
original_frame = original_data[:, :samples_per_frame].T  # 形状为 (1024, 2)

# 绘图对比
plt.figure(figsize=(12, 6))
for ch in range(nchan):
    plt.subplot(nchan, 1, ch + 1)
    plt.plot(original_frame[:, ch], label=f'Original Ch{ch}', linestyle='--', color='blue')
    plt.plot(decoded_data[:, ch], label=f'Decoded Ch{ch}', linestyle='-', color='orange')
    plt.title(f'Channel {ch}: Original vs Decoded (2-bit real)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
plt.tight_layout()
plt.show()
