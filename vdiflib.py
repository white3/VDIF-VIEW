import copy
import threading
import vdifheader as vh
import numpy as np
import threading

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
        stats['CHANNELS_BAND_MHz'] = (header.data_frame_number+1) * (header.data_frame_length - vh.VDIF_HEADER_BYTES) * 8.0 / header.bits_per_sample / header.num_channels / 1000000.0 / 2
        for k, v in header.to_dict.items():
            stats[k.name] = v
        stats['DATA_FRAME_NUMBER'] += 1
        return stats
    else:
        header = vh.get_first_header(filepath)
        for k, v in header.to_dict.items():
            stats[k.name] = v
        return stats

def read_vdif_frame(f, channel=1, vtype='real', count=None):
    result = [[]]*channel
    for h, b in vh.get_VDIFs(f, count):
        # print(h.data_frame_number, h.data_frame_length, h.bits_per_sample, h.num_channels)
        if h.invalid_flag:
            continue
        data = list(decode_quantized_samples(b, h.bits_per_sample))
        if vtype == 'complex':
            continue
        elif vtype =='real':
            for i in range(channel):
                result[i]+=data[i::channel]
    return result

def read_vdif_frame_by_input(f, channel=1, vtype='real', bits=2, count=None):
    result = [[]]*channel
    for h, b in vh.get_VDIFs(f, count):
        if b is None:
            return None
        # print(h.data_frame_number, h.data_frame_length, h.bits_per_sample, h.num_channels)
        if h.invalid_flag:
            continue
        data = list(decode_quantized_samples(b, bits))
        if vtype == 'complex':
            for i in range(channel):
                result[i]+=np.array(data[i::channel] + 1j*data[1::channel])
        elif vtype =='real':
            for i in range(channel):
                result[i]+=data[i::channel]
    return result

def vdif_config2str(vdif_config):
    if 'bandwidth' in vdif_config and 'channels' in vdif_config and 'bits' in vdif_config:
        if 'threads' in vdif_config:
            return f"{vdif_config['bandwidth']}-{vdif_config['channels']}-{vdif_config['bits']}-{vdif_config['threads']}"
        else:
            return f"{vdif_config['bandwidth']}-{vdif_config['channels']}-{vdif_config['bits']}"
    else:
        return None 

def vdif_config2str(vdifstr='512-16-2'):
    tmp = vdifstr.strip().split('-')
    if len(tmp) == 3:
        return {
            'bandwidth': int(tmp[0]),
            'channels': float(tmp[1]),
            'bits': int(tmp[2]),
            'threads': 1
        }
    elif len(tmp) == 4:
        return {
            'bandwidth': int(tmp[0]),
            'channels': float(tmp[1]),
            'bits': int(tmp[2]),
            'threads': int(tmp[3])
        }
    else:
        return None

def parse_vdif_config(vdifstr='512-16-2'):
    tmp = vdifstr.strip().split('-')
    if len(tmp) == 3:
        return {
            'bandwidth': float(tmp[0]),
            'channels': int(tmp[1]),
            'bits': int(tmp[2]),
            'threads': 1
        }
    elif len(tmp) == 4:
        return {
            'bandwidth': float(tmp[0]),
            'channels': int(tmp[1]),
            'bits': int(tmp[2]),
            'threads': int(tmp[3])
        }
    else:
        return None

class VDIFProcessThread(threading.Thread):
    def __init__(self, vdifstr, integration, fftsize, stats, vdif_path, tmp_data, parent=None):
        threading.Thread.__init__(self)
        self.proc_params = parse_vdif_config(vdifstr=vdifstr)
        self.integration = integration
        self.fftsize = fftsize
        self.stats = copy.deepcopy(stats)
        self.stats['running'] = True
        self.vdif_path = vdif_path
        self.tmp_data = tmp_data
        self.parent = parent
        self.stats['lock'] = threading.Lock()

    def stopProcess(self):
        with self.stats['lock']:
            self.stats['running'] = False

    def isProcessAlive(self):
        with self.stats['lock']:
            return self.stats['running']

    def run(self):
        print(f"Processing {self.vdif_path} with {self.proc_params}")
        integration = self.integration
        frames = integration * self.proc_params['bandwidth'] * 2 \
            *self.proc_params['bits']/8*1000000 \
            /(self.stats['DATA_FRAME_LENGTH']-vh.VDIF_HEADER_BYTES)
        if self.stats['DATA_TYPE'] == 'complex':
            frames *= 2
        frames = int(frames)
        # 1秒的字节数
        # self.proc_params['channels']*self.proc_params['bandwidth']*self.proc_params['bits']
        fftsize = self.fftsize
        print(f"Processing {frames} frames with {fftsize} FFT size")

        freq = np.linspace(0, 
            self.proc_params['bandwidth'], 
            int(fftsize*self.proc_params['channels']//2))
            
        idx = 0
        with open(self.vdif_path, 'rb') as f:
            while self.isProcessAlive():
                print(f"Processing batch {idx+1}/{frames}")
                data = read_vdif_frame_by_input(f=f, 
                    channel=self.proc_params['channels'], 
                    vtype=self.stats['DATA_TYPE'], 
                    bits=self.proc_params['bits'], 
                    count=frames)
                if data is None or len(data) == 0:
                    break
                output = []
                print(f"Processing batch {idx+1}/{frames} - FFT")
                for ichan in range(self.proc_params['channels']):
                    spectrum = np.zeros(fftsize//2, dtype=np.complex64)
                    for i in range(0, len(data[ichan]), fftsize):
                        temp = data[ichan][i:i+fftsize]
                        spectrum_tmp = np.fft.fft(temp)[:fftsize//2]
                        spectrum += spectrum_tmp / (fftsize//2)
                    output.append(spectrum)
                # np.save(out_path, np.array(output))
                self.tmp_data.append([freq, np.concatenate(output, axis=0)])
                if idx == 0:
                    self.parent.plot_current_frame()
                self.parent.plotnum.increment()
                idx += 1

if __name__ == '__main__':
    vdif_path = './data/testpeb1_01min.vdif'
    vdif_path="../gpu_pulsar_pipeline/res/a2102gt6.vdif"
    # stats = analyze_vdif_file(vdif_path)
    # print(stats)

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

    
