import copy
import threading
import struct
import vdifheader as vh
import numpy as np
# Lookup tables for 1-bit and 2-bit quantization (example mappings)
ONE_BIT_MAP = {0: -1.0, 1: 1.0}
TWO_BIT_MAP = {
    0b00: -3.0,
    0b01: -1.0,
    0b10: 1.0,
    0b11: 3.0
}

import threading

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

def decode_quantized_samples(payload: bytes, bit_depth: int):
    if bit_depth == 1:
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        mapped = np.vectorize(ONE_BIT_MAP.get)(bits)
        return mapped
    elif bit_depth == 2:
        bytes_arr = np.frombuffer(payload, dtype=np.uint8)
        samples = np.zeros(len(bytes_arr) * 4, dtype=np.float32)
        for i, b in enumerate(bytes_arr):
            for j in range(4):
                bits = (b >> (6 - 2 * j)) & 0b11
                samples[i * 4 + j] = TWO_BIT_MAP[bits]
        return samples
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
        return header.to_dict

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

class VDIFProcessThread(threading.Thread):
    def __init__(self, integration, fftsize, stats, vdif_path, tmp_data, parent=None):
        threading.Thread.__init__(self)
        self.proc_params = {
            'integration': 0.1,
            'fftsize': 1024,
            'filepath': None,
            'running': False
        }
        self.integration = integration
        self.fftsize = fftsize
        self.stats = copy.deepcopy(stats)
        self.stats['running'] = True
        self.vdif_path = vdif_path
        self.tmp_data = tmp_data
        self.parent = parent

    def run(self):
        integration = self.integration
        frames = integration * self.stats['DATA_FRAME_NUMBER']
        fftsize = self.fftsize
        print(f"Processing {frames} frames with {fftsize} FFT size")

        freq = np.linspace(0, 
            self.stats['CHANNELS_BAND_MHz'] * self.stats['NUM_CHANNELS'], fftsize//2)
            
        idx = 0
        with open(self.vdif_path, 'rb') as f:
            while self.stats['running']:
                print(f"Processing batch {idx+1}/{frames}")
                data = read_vdif_frame_by_input(f=f, 
                    channel=self.stats['NUM_CHANNELS'], 
                    vtype=self.stats['DATA_TYPE'], 
                    bits=self.stats['BITS_PER_SAMPLE'], 
                    count=frames)
                output = []
                output.append(freq)
                print(f"Processing batch {idx+1}/{frames} - FFT")
                for ichan in range(self.stats['NUM_CHANNELS']):
                    spectrum = np.zeros(fftsize//2, dtype=np.complex64)
                    for i in range(0, len(data[ichan]), fftsize):
                        temp = data[ichan][i:i+fftsize]
                        spectrum_tmp = np.fft.fft(temp)[:fftsize//2]
                        spectrum += spectrum_tmp / (len(data[ichan])//fftsize//2)
                    output.append(spectrum)
                # np.save(out_path, np.array(output))
                self.tmp_data.append(output)
                if idx == 0:
                    self.parent.plot_current_frame()
                self.parent.plotnum.increment()
                idx += 1

if __name__ == '__main__':
    vdif_path = './data/testpeb1_01min.vdif'
    # stats = analyze_vdif_file(vdif_path)
    # print(stats)

    with open(vdif_path, 'rb') as f:
        data = np.array(read_vdif_frame(f, 2, 'real', 4))
        print(data.shape)

    