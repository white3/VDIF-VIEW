import copy
import threading
import vdifheader as vh
import numpy as np
import threading
from datetime import datetime, timedelta
import math

# Lookup tables for 1-bit and 2-bit quantization (example mappings)
ONE_BIT_MAP = {0: -1.0, 1: 1.0}
TWO_BIT_MAP = {
    0b00: -3.3359,
    0b01: -1.0,
    0b10: 1.0,
    0b11: 3.3359
}

class AtomicInt:
    def __init__(self, initial=0):
        self.value = initial
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self.value += 1

    def get(self):
        with self._lock:
            return self.value

# 定义映射级别
LEVELS = np.array([-3.3359, -1.0, 1.0, 3.3359], dtype=np.float32)

# 构建查找表（LUT）
LUT = np.empty((256, 4), dtype=np.float32)
for i in range(256):
    for j in range(4):
        k = (i >> (2 * j)) & 0x3
        LUT[i, j] = LEVELS[k]

def convert2tofloat(src: bytes) -> np.ndarray:
    """
    将 2-bit 量化的字节数据解码为浮点数数组。

    参数：
        src (bytes): 输入的字节数据，每个字节包含 4 个 2-bit 的样本。

    返回：
        np.ndarray: 解码后的浮点数数组。
    """
    src_array = np.frombuffer(src, dtype=np.uint8)
    dest = LUT[src_array].reshape(-1)
    return dest

def decode_2bit_samples(payload: bytes, bitorder: str):
    """
    解码 2-bit 量化数据。

    参数：
        payload (bytes): 原始字节数据。
        bitorder (str): 位顺序，'big' 或 'little'。

    返回：
        np.ndarray: 解码后的浮点数数组。
    """
    bytes_arr = np.frombuffer(payload, dtype=np.uint8)
    samples = np.zeros(len(bytes_arr) * 4, dtype=np.float32)
    for i, b in enumerate(bytes_arr):
        for j in range(4):
            if bitorder == 'big':
                bits = (b >> (6 - 2 * j)) & 0b11
            else:
                bits = (b >> (2 * j)) & 0b11
            samples[i * 4 + j] = TWO_BIT_MAP[bits]
    return samples

def decode_quantized_samples(payload: bytes, bit_depth: int, bitorder='small'):
    if bit_depth == 1:
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder=bitorder)
        mapped = np.vectorize(ONE_BIT_MAP.get)(bits)
        return mapped
    elif bit_depth == 2:
        return convert2tofloat(payload)
    else:
        samples = np.frombuffer(payload, dtype=np.int8).astype(np.float32)
        return samples

def analyze_vdif_file(filepath):
    stats = {}
    header = vh.get_headers_second(filepath)
    if header:
        # 总帧数 * 帧字节数 / 比特位数 / 通道数 / 1e6 = MHz
        stats['CHANNELS_BAND_MHz'] = (header.data_frame_number+1) * \
            (header.data_frame_length - vh.VDIF_HEADER_BYTES) * 8.0 \
            / header.bits_per_sample / header.num_channels / 1000000.0 / 2
        if header.data_type == 'complex':
            stats['CHANNELS_BAND_MHz'] /= 2
        stats['BPS_MHz'] = (header.data_frame_number+1) * \
            (header.data_frame_length - vh.VDIF_HEADER_BYTES) \
             * 8.0 / 1000000.0
        for k, v in header.to_dict.items():
            stats[k.name] = v
        stats['DATA_FRAME_NUMBER'] += 1
    else:
        header = vh.get_first_header(filepath)
        for k, v in header.to_dict.items():
            stats[k.name] = v

    header0 = vh.get_first_header(filepath)
    seconds = header0.data_frame_number / stats['DATA_FRAME_NUMBER']
    stats['First Frame'] = header0.get_timestamp() + \
        timedelta(seconds=seconds)
    return stats 

def parse_vdif_config(vdifstr='8000-512-16-2'):
    tmp = vdifstr.strip().split('-')
    if len(tmp) == 4:
        vdifconf =  {
            'fbodybytes': int(tmp[0]), 
            'bps': float(tmp[1]),
            'channels': int(tmp[2]),
            'bits': int(tmp[3]),
            'threads': 1
        }
        vdifconf['bandwidth'] = vdifconf['bps'] / vdifconf['bits'] / 2
    elif len(tmp) == 5:
        vdifconf =  {
            'fbodybytes': int(tmp[0]), 
            'bps': float(tmp[1]),
            'channels': int(tmp[2]),
            'bits': int(tmp[3]),
            'threads': int(tmp[4])
        }
        vdifconf['bandwidth'] = vdifconf['bps'] / vdifconf['bits'] / 2
    else:
        return None
    return vdifconf

class VDIFProcessThread(threading.Thread):
    def __init__(self, vdifstr, fftsize, stats, vdif_path, qt_queue, parent=None):
        threading.Thread.__init__(self)
        self.proc_params = parse_vdif_config(vdifstr=vdifstr)
        self.fftsize = fftsize
        self.stats = copy.deepcopy(stats)
        self.stats['running'] = True
        self.vdif_path = vdif_path
        self.qt_queue = qt_queue
        self.parent = parent
        self.output = {
            'Last Frame': None,
            'frames': 0,
            'err_frames': 0,
        }
        self.stats['lock'] = threading.Lock()

    def stopProcess(self):
        with self.stats['lock']:
            self.stats['running'] = False

    def isProcessAlive(self):
        with self.stats['lock']:
            return self.stats['running']
        
    def getFreq(self):
        nchan = self.proc_params['channels']    
        bandw = self.proc_params['bandwidth']  
        fftsize = self.fftsize 
        freq = []
        for ifft in range(nchan):
            freq.append(np.linspace(bandw/nchan*ifft, 
                bandw/nchan*(ifft+1), int(fftsize//2)))
        return np.array(freq)

    def run(self):
        print(f"Processing {self.vdif_path} with {self.proc_params}")

        nbits = self.proc_params['bits']
        nchan = self.proc_params['channels']
        vtype = self.stats['DATA_TYPE']
        fbodybytes = self.proc_params['fbodybytes']
        fftsize = self.fftsize
        
        if vtype == 'complex':
            fbodynum = fbodybytes * 4 // nbits
        else:
            fbodynum = fbodybytes * 8 // nbits
        # 依次循环FFT数量
        nfbin = fbodynum // nchan
        nfft = abs(fftsize * nfbin) // math.gcd(fftsize, nfbin) 
        nfft = nfft // fftsize
        # 一次循环读取数据数量
        nframes = nfft * fftsize // nfbin
        print(f"Processing {nframes} frames with {fftsize} points for {nfft} FFTs in each loop")

        idx = 0
        # 一直积分到结束
        with open(self.vdif_path, 'rb') as fvdif:
            while self.isProcessAlive():
                # read nframes frames that bodysize is fbodybytes by filehandle fvdif
                # print(f"Reading {nframes} frames from {self.vdif_path}")
                fhead01, rframes, raframes, vdatas = read_vdif(fvdif, nframes, fbodybytes)
                self.output['frames'] += rframes
                self.output['err_frames'] += raframes - rframes
                if rframes != nframes:
                    print(f"Warning: {rframes} frames read, but {nframes} expected")
                    break
                vdatas_float = decode_quantized_samples(vdatas, nbits)

                # channel by channel
                # print(f"Processing batch {idx+1} ({nframes} frames) - Channel")
                # 初始化 vdata_chaned 为列表长度 nfft，每个元素是 nchan 个空列表
                vdata_chaned = [[None for _ in range(nchan)] for _ in range(nfft)]

                if vtype == 'complex':
                    for ifft in range(nfft):
                        for ichan in range(nchan):
                            start = ifft * fftsize
                            end = start + fftsize
                            real_part = np.array(vdatas_float[ichan::nchan][start:end])
                            imag_part = np.array(vdatas_float[1::nchan][start:end])
                            vdata_chaned[ifft][ichan] = real_part + 1j * imag_part
                        # FFT, vdata_chaned: [nchan][nfft*fftsize]
                        # print(f"Processing batch {idx+1} ({nframes} frames / {ifft} FFTs / Total {nfft} FFTs) - FFT")
                        fft_result = np.fft.fft(vdata_chaned[ifft], axis=-1)
                        fft_amp = np.abs(fft_result) / nfft
                        fft_phase = np.angle(fft_result)
                        # 形状 (nchan, fftsize)
                        # print(f"Put {fftsize} FFT data to queue")
                        self.qt_queue.put(
                            [fft_amp[:,:fftsize//2], fft_phase[:,:fftsize//2]])
                elif vtype == 'real':
                    for ifft in range(nfft):
                        for ichan in range(nchan):
                            # 按照你的填充方式，取第 i fft 块里 ichan 通道的 fftsize 数据
                            # 计算起止索引
                            start = ifft * fftsize
                            end = start + fftsize
                            vdata_chaned[ifft][ichan] = np.array(
                                vdatas_float[ichan::nchan][start:end])
                        
                        # FFT, vdata_chaned: [nchan][nfft*fftsize]
                        # print(f"Processing batch {idx+1} ({nframes} frames / {ifft} FFTs / Total {nfft} FFTs) - FFT")
                        fft_result = np.fft.fft(vdata_chaned[ifft], axis=-1)
                        fft_amp = np.abs(fft_result) / nfft
                        fft_phase = np.angle(fft_result)
                        # 形状 (nchan, fftsize)
                        # print(f"Put {fftsize} FFT data to queue")
                        self.qt_queue.put(
                            [fft_amp[:,:fftsize//2], fft_phase[:,:fftsize//2]])
                idx += 1
                last_frame = fhead01[1]
        
        seconds = last_frame.data_frame_number / self.stats['DATA_FRAME_NUMBER']
        self.output['Last Frame'] = last_frame.get_timestamp() + \
            timedelta(seconds=seconds)
        self.output['Duration (seconds)'] = (self.output['Last Frame'] - self.stats['First Frame']).total_seconds()
        self.parent.update_stats(self.output)

def power_to_db(power, ref=1.0, floor_db=-np.inf):
    """
    将普通功率值转换为分贝(dB)。

    参数：
    - power: 输入功率值，标量或数组，必须非负。
    - ref: 参考功率，默认为 1.0。
    - floor_db: 最小dB值，用于限制结果下限，默认不限制。

    返回：
    - dB值，float或ndarray，与输入power同形状。
    """
    power = np.array(power, dtype=float)
    # 避免对0取对数，设一个极小值替代0
    power_clipped = np.clip(power, 1e-20, None)
    db = 10 * np.log10(power_clipped / ref)
    if floor_db != -np.inf:
        db = np.maximum(db, floor_db)
    return db

def read_vdif(f, count, fbodybytes):
    fhead_bytes = f.read(vh.VDIF_HEADER_BYTES)
    try:
        tmp = vh.VDIFHeader.parse(fhead_bytes)
    except:
        return None, 0, 0, b''
    fhead01 = [tmp]
    rframes = 0
    vdatas = b''
    icount = 0
    while True:
        try:
            fhead = vh.VDIFHeader.parse(fhead_bytes)
            flag = True
        except:
            flag = False
        fbody = f.read(fbodybytes)
        if len(fbody)!= fbodybytes:
            break
        icount += 1
        if flag and not fhead.invalid_flag:
            vdata = fbody
            vdatas+=vdata
            rframes += 1
        if rframes == count:
            break
        fhead_bytes = f.read(vh.VDIF_HEADER_BYTES)
        if len(fhead_bytes)!= vh.VDIF_HEADER_BYTES:
            break
    fhead01.append(fhead)
    return fhead01, rframes, icount, vdatas

if __name__ == '__main__':
    vdif_path = './data/testpeb1_01min.vdif'
    vdif_path="../gpu_pulsar_pipeline/res/a2102gt6.vdif"
    # stats = analyze_vdif_file(vdif_path)

    with open(vdif_path, 'rb') as f:
        r = vh.get_VDIFs(f)
        h, b = next(r)
        converted = convert2tofloat(b)
        big_data = decode_2bit_samples(b, 'big')
        little_data = decode_2bit_samples(b, 'little')
    
    for i in range(32):
        print(converted[i*4+0], converted[i*4+1], converted[i*4+2], converted[i*4+3], end='|')
        if (i+1) % 4 == 0:
            print()

    
