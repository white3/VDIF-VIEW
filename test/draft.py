import numpy as np
a=np.linspace(1, 128, 128)
b=[]
for i in range(4):
    b.append(a[i::4])

c = np.fft.fft(b)

d = []
for i in range(4):
    d.append(np.fft.fft(b[i]))

e = c-d
for i in range(4):
    print(np.abs(e[i])[:5])



import numpy as np

nchan, nfft, fftsize = 2, 4, 16
vdata_chaned = np.array([
    np.linspace(0, nfft*fftsize-1, nfft*fftsize),
    np.linspace(1, nfft*fftsize, nfft*fftsize)])

# 假设 vdata_chaned 是 numpy.ndarray，形状 (nchan, nfft * fftsize)
# 先 reshape 成 (nchan, nfft, fftsize)
vdata_reshaped = vdata_chaned.reshape((nchan, nfft, fftsize))

# 对每个通道的每段做 FFT，axis=-1 指对最后一个维度做 fft
fft_result = np.fft.fft(vdata_reshaped, axis=-1)  # 形状 (nchan, nfft, fftsize)

# 沿 nfft 维度累加，得到每个通道的累积 fft 结果
accum_fft = np.sum(fft_result, axis=1)  # 形状 (nchan, fftsize)

# accum_fft 即为最终结果