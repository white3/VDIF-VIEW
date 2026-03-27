import copy
import sys
import os
import numpy as np
import threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QSpinBox,
    QLabel, QFileDialog, QHBoxLayout, QDoubleSpinBox, QLineEdit,
    QTableView, QHeaderView, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import (QStandardItemModel, QStandardItem, QIcon)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import vdiflib
from datetime import datetime, timezone
#from qt_material import apply_stylesheet
#import qdarktheme
import queue
from PyQt5.QtWidgets import QSplitter

class VDIFViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VDIF Viewer - With Stats Sidebar")
        self.resize(1480, 800)
        self.vdif_path = None
        # data cache for background processing
        self.vdifqueue = queue.Queue()
        self.tmp_data = []
        self.freq = None
        # data cache for plotting
        self.plotnum = vdiflib.AtomicInt(-1)
        # data cache for processing parameters
        self.stats = {'CHANNELS_BAND': '32.00 MHz', 'INVALID_FLAG': False, 'LEGACY_MODE': False, 'REFERENCE_EPOCH': datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc), 'SECONDS_FROM_EPOCH': 1155000, 'UNASSIGNED_FIELD': 0, 'DATA_FRAME_NUMBER': 1999, 'VDIF_VERSION': 0, 'NUM_CHANNELS': 1, 'DATA_FRAME_LENGTH': 8032, 'BITS_PER_SAMPLE': 2, 'THREAD_ID': 0, 'EXTENDED_DATA_VERSION': 0, 'DATA_TYPE': 'real', 'STATION_ID': 'NA', 'EXTENDED_DATA': {}}
        self.prcthread = None
        self.plotthread = PlotUpdateThread(self.vdifqueue, self)
        self.plotthread.start()
        self.ready2plot = False
        self.fleshPeriod = 0.5
        self.vdif_config = {
            'bandwidth': 512.0,
            'channels': 16,
            'bits': 2,
            'threads': 1
        }
        self.title_suffix = None
        self.thread_channels = {}   # {thread_id: channel_count}
        self.thread_ids = []        # sorted thread ids
        self.init_ui()

    def _stop_processing_thread(self):
        """Stop current processing thread and wait briefly for exit."""
        if self.prcthread and self.prcthread.is_alive():
            self.prcthread.stopProcess()
            self.prcthread.join(timeout=1.5)

    def _clear_plot_queue(self):
        """Drain queued FFT frames to avoid stale-file data mixing."""
        while True:
            try:
                self.vdifqueue.get_nowait()
            except queue.Empty:
                break

    def closeEvent(self, event):
        print("Closing VDIF Viewer")
        self._stop_processing_thread()
        self.plotthread.stop()
        return super().closeEvent(event)

    def init_ui(self):
        self.setWindowIcon(QIcon('./icon.png'))
        main_layout = QVBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        int_layout = QHBoxLayout()
        file_label_layout = QHBoxLayout()

        self.VDIF_settings_label = QLabel("VDIF Settings:")
        self.VDIF_settings_label.setFixedWidth(100)
        self.VDIF_settings_combo = QLineEdit()
        self.VDIF_settings_combo.setPlaceholderText("8000-512-16-2")
        self.VDIF_settings_combo.setEnabled(False)
        self.VDIF_settings_combo.setFixedWidth(125)
        self.VDIF_settings_checkbox = QCheckBox()
        self.VDIF_settings_checkbox.setChecked(False)
        self.VDIF_settings_checkbox.setFixedWidth(150)
        self.VDIF_settings_checkbox.setText("Enable Settings")
        self.VDIF_settings_checkbox.toggled.connect(
            lambda checked: self.VDIF_settings_combo.setEnabled(checked)
        )
        self.file_label = QLabel("No file selected")

        self.file_button = QPushButton("Select VDIF File")
        self.file_button.clicked.connect(self.select_file)
        self.file_button.setFixedWidth(200)

        self.FFT_size_label = QLabel("FFT Size:")
        self.FFT_size_label.setFixedWidth(100)
        self.FFT_size_spin = QSpinBox()
        self.FFT_size_spin.setFixedWidth(125)
        self.FFT_size_spin.setMinimum(32)
        self.FFT_size_spin.setMaximum(65536)
        self.FFT_size_spin.setValue(2048)
        
        self.start_button = QPushButton("Plot")
        self.start_button.setEnabled(False)
        self.start_button.setFixedWidth(200)
        self.start_button.clicked.connect(self.start_background_processing)

        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        self.figure.clear()
        self.set_frame("Please select a VDIF file first.")

        self.current_channel_label = QLabel("Cur Chan:")
        self.current_channel_label.setFixedWidth(100)
        self.channel_spin = QSpinBox()
        self.channel_spin.setMinimum(-1)
        self.channel_spin.setValue(0)
        self.channel_spin.valueChanged.connect(self.plot_current_frame)
        self.channel_spin.setFixedWidth(125)

        self.cur_thread_label = QLabel("Cur Thread:")
        self.cur_thread_label.setFixedWidth(100)
        self.thread_spin = QSpinBox()
        self.thread_spin.setMinimum(-1)
        self.thread_spin.setValue(0)
        self.thread_spin.valueChanged.connect(self.on_thread_changed)
        self.thread_spin.setFixedWidth(125)

        self.reflush_rate_label = QLabel("RR:")
        self.reflush_rate_label.setFixedWidth(50)
        self.reflush_rate_spin = QDoubleSpinBox()
        self.reflush_rate_spin.setMinimum(0.1)
        self.reflush_rate_spin.setValue(self.fleshPeriod)
        self.reflush_rate_spin.valueChanged.connect(self.update_flesh_period)
        self.reflush_rate_spin.setFixedWidth(100)

        self.center_freq_label = QLabel("Center Freq (MHz):")
        self.center_freq_label.setFixedWidth(130)
        self.center_freq_input = QDoubleSpinBox()
        self.center_freq_input.setMinimum(0)
        self.center_freq_input.setMaximum(1e9)
        self.center_freq_input.setDecimals(3)
        self.center_freq_input.setSingleStep(1.0)
        self.center_freq_input.setValue(0.0)  # 默认0表示无偏移
        self.center_freq_input.setFixedWidth(150)


        self.reduce_label = QLabel("View Max Amp:")
        self.reduce_label.setFixedWidth(100)
        self.reduce_spin = QDoubleSpinBox()
        self.reduce_spin.setMinimum(-1)
        self.reduce_spin.setMaximum(np.inf)
        self.reduce_spin.setValue(-1)
        self.reduce_spin.valueChanged.connect(self.plot_current_frame)
        self.reduce_spin.setFixedWidth(125)

        file_label_layout.addWidget(self.file_button)
        file_label_layout.addWidget(self.VDIF_settings_label)
        file_label_layout.addWidget(self.VDIF_settings_combo)
        file_label_layout.addWidget(self.VDIF_settings_checkbox)
        file_label_layout.addWidget(self.file_label)
        left_layout.addLayout(file_label_layout)
        
        int_layout.addWidget(self.center_freq_label)
        int_layout.addWidget(self.center_freq_input)

        int_layout.addWidget(self.start_button)
        int_layout.addWidget(self.FFT_size_label)
        int_layout.addWidget(self.FFT_size_spin)
        int_layout.addWidget(self.reduce_label)
        int_layout.addWidget(self.reduce_spin)
        int_layout.addWidget(self.current_channel_label)
        int_layout.addWidget(self.channel_spin)
        int_layout.addWidget(self.cur_thread_label)
        int_layout.addWidget(self.thread_spin)
        int_layout.addWidget(self.reflush_rate_label)
        int_layout.addWidget(self.reflush_rate_spin)
        int_layout.setAlignment(Qt.AlignLeft)
        left_layout.addLayout(int_layout)

        # left_layout.addLayout(nav_layout)
        left_layout.addWidget(self.canvas, stretch=1)

        self.model = QStandardItemModel(15, 2)
        self.model.setHorizontalHeaderLabels(["VDIFHeaderField", "value"])

        self.tableview = QTableView()
        self.tableview.setAlternatingRowColors(True)

        # 设置模型
        self.tableview.setModel(self.model)

        # 设置列宽按比例填充（60% / 40%）
        header = self.tableview.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)

        # 可选：设置比例（用 setStretchLastSection 和 resizeSection）
        header.setStretchLastSection(False)
        total_width = self.tableview.viewport().width()
        header.resizeSection(0, int(total_width * 0.6))
        header.resizeSection(1, int(total_width * 0.4))

        # 添加到右侧布局（自动填充宽度）
        right_layout.addWidget(self.tableview)
        
        splitter = QSplitter()
        left_widget = QWidget(); left_widget.setLayout(left_layout)
        right_widget = QWidget(); right_widget.setLayout(right_layout)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        # 加入主布局
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        self.start_button.setEnabled(False)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 
            "Open VDIF File", "", "VDIF Files (*.vdif);;All Files (*)")
        if path:
            # 切换文件前先停旧处理并清空旧数据，避免新旧文件数据混流
            self._stop_processing_thread()
            self._clear_plot_queue()
            self.tmp_data = []
            self.ready2plot = False
            self.title_suffix = None

            self.file_label.setText(f"Selected: {path}")
            self.vdif_path = path
            self.stats = vdiflib.analyze_vdif_file(path)
            if self.stats.get("THREAD_CHANNELS"):
                threads_ch = self.stats["THREAD_CHANNELS"]
                chan_counts = list(threads_ch.values())
                if all(c == chan_counts[0] for c in chan_counts):
                    # 所有 thread 通道数相同
                    self.stats["├─ Thread CH"] = chan_counts[0]
                else:
                    # 通道数不同，分别列出
                    for tid, ch in sorted(threads_ch.items()):
                        self.stats[f"├─ Thread {tid}"] = ch
            self.display_stats(self.stats)
            self.ready2plot = True
                        # --- multi-thread meta from stats ---
            self.thread_channels = self.stats.get('THREAD_CHANNELS', {})
            self.thread_ids = sorted(self.thread_channels.keys())

            # 线程选择范围：[-1, max_tid]；若没有多线程，保持[-1, -1]
            if self.thread_ids:
                self.thread_spin.setMinimum(-1)
                self.thread_spin.setMaximum(max(self.thread_ids))
            else:
                self.thread_spin.setMinimum(-1)
                self.thread_spin.setMaximum(-1)

            # 当选择了具体 thread（见 on_thread_changed），channel 的最大值会被动态重设
            # 这里先跟随 vdif_config 的 channels 作为兜底
            self.channel_spin.setMaximum(int(self.vdif_config.get('channels', 1)) - 1)
            self.channel_spin.blockSignals(True)
            self.thread_spin.blockSignals(True)
            self.channel_spin.setValue(0)
            self.thread_spin.setValue(-1)
            self.channel_spin.blockSignals(False)
            self.thread_spin.blockSignals(False)

            if self.stats.get('CHANNELS_BAND_MHz'):
                vdifstr = "{:d}-{:d}-{:d}-{:d}".format(
                        int(self.stats['DATA_FRAME_LENGTH']-vdiflib.vh.VDIF_HEADER_BYTES), 
                        int(self.stats['BPS_MHz']), 
                        int(self.stats['NUM_CHANNELS']), 
                        int(self.stats['BITS_PER_SAMPLE']), 
                    )
                self.vdif_config = vdiflib.parse_vdif_config(vdifstr)
                self.VDIF_settings_combo.setText(vdifstr)
            else:
                self.alert(title="Error", text="Please input the VDIF settings manually: \n8000-512-16-2\n<VDIF Body Length in bytes>-<bps in MHz>-<num channels>-<bits per sample>")
                self.VDIF_settings_checkbox.setChecked(True)
            self.start_button.setEnabled(True)

    def display_stats(self, stats):
        i = 0
        for field_name, field_value in stats.items():
            self.model.setItem(i, 0, QStandardItem(str(field_name)))
            self.model.setItem(i, 1, QStandardItem(str(field_value)))
            i += 1

        header = self.tableview.horizontalHeader()
        # 第0列自适应内容
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  
        # 第1列填满剩余空间
        header.setSectionResizeMode(1, QHeaderView.Stretch)           
    
    def update_flesh_period(self):
        self.fleshPeriod = self.reflush_rate_spin.value()

    # ---------- helpers for multi-thread UI ----------

    def _valid_channel_range_for_thread(self, tid: int):
        """返回指定 thread 的合法 channel 范围 (start=0, end=cnt-1)。不存在则返回 None。"""
        if tid in self.thread_channels:
            cnt = int(self.thread_channels[tid])
            return (0, cnt - 1)
        return None

    def _overall_thread_bounds(self):
        """返回 (t_first, t_end)。若无多线程，返回(None, None)。"""
        if not self.thread_ids:
            return (None, None)
        return (self.thread_ids[0], self.thread_ids[-1])

    def _overall_channel_bounds(self, tid: int):
        """返回某个thread的(0, end)；若tid为None则返回全局最大通道范围(0, max_end)。"""
        if tid is not None:
            rng = self._valid_channel_range_for_thread(tid)
            if rng:
                return rng
            return (0, -1)
        # 全局：取所有thread的最大通道数
        if not self.thread_channels:
            return (0, int(self.vdif_config.get('channels', 1)) - 1)
        max_cnt = max(int(v) for v in self.thread_channels.values())
        return (0, max_cnt - 1)

    def on_thread_changed(self, tid_val: int):
        """thread_spin 调整时：动态更新 channel 的范围 & 必要时给出警告"""
        # 当选择具体 thread 时，重设 channel 的上限为该 thread 的通道数-1
        if tid_val != -1 and self.thread_channels:
            rng = self._valid_channel_range_for_thread(tid_val)
            if rng is None:
                # 选择了不存在的 thread id，回退到 -1 并警告
                self.thread_spin.blockSignals(True)
                self.thread_spin.setValue(-1)
                self.thread_spin.blockSignals(False)
                self.alert(title="Warning",
                           text=f"Thread {tid_val} 不存在于数据中。已切回所有线程（-1）。")
            else:
                # 更新 channel 上限
                _, ch_end = rng
                self.channel_spin.setMaximum(max(0, ch_end))
                # 若当前 channel 超出范围，警告并回退到 -1
                if self.channel_spin.value() > ch_end:
                    self.channel_spin.setValue(-1)
                    self.alert(title="Warning",
                               text=f"Thread {tid_val} 的通道范围为 0..{ch_end}，已将通道切回 -1（全部）。")
        else:
            # 选择 “全部线程(-1)” 时，channel 上限设置为全局最大
            _, ch_end = self._overall_channel_bounds(None)
            self.channel_spin.setMaximum(max(0, ch_end))

        # 触发重绘与标题更新
        self.plot_current_frame()
    
    # ---------- helpers for multi-thread UI/end ----------

    def alert(self, title="Error", text="The VDIF data duration should exceed 1 second!"):
        dlg = QMessageBox(self)
        dlg.setWindowTitle(title)
        dlg.setText(text)
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.setIcon(QMessageBox.Critical)
        dlg.exec()

    def start_background_processing(self):
        self.start_button.setEnabled(False)
        # 重启处理前先确保旧线程退出并清空旧队列数据
        self._stop_processing_thread()
        self._clear_plot_queue()
        self.plotnum = vdiflib.AtomicInt(-1)
        self.fftsize = self.FFT_size_spin.value()
        self.prcthread = vdiflib.VDIFProcessThread(
            vdifstr=self.VDIF_settings_combo.text(),
            fftsize=self.fftsize,
            stats=self.stats,
            vdif_path=self.vdif_path,
            qt_queue=self.vdifqueue,
            parent=self
        )
        self.vdif_config = vdiflib.parse_vdif_config(self.VDIF_settings_combo.text())
        self.channel_spin.setMaximum(int(self.vdif_config['channels']-1))
        self.last_plot_time = datetime.now().timestamp()
        if self.ready2plot:
            self.freq = self.prcthread.getFreq()
            if self.stats.get('DATA_TYPE') == 'complex':
                self.tmp_data = [
                    np.zeros((self.vdif_config['channels'], self.fftsize), dtype=float),     # full spectrum amplitude
                    np.zeros((self.vdif_config['channels'], self.fftsize), dtype=complex)    # full spectrum phase
                ]
            else:  # real
                self.tmp_data = [
                    np.zeros((self.vdif_config['channels'], self.fftsize // 2), dtype=float),
                    np.zeros((self.vdif_config['channels'], self.fftsize // 2), dtype=complex)
                ]
            self.prcthread.start()
            self.ready2plot = False

    def update_stats(self, stats):
        self.stats.update(stats)
        self.display_stats(self.stats)

    def update_data(self, data):
        if not self.tmp_data:
            return
        # 避免切文件竞态导致旧shape数据写入新缓存
        if data[0].shape != self.tmp_data[0].shape or data[1].shape != self.tmp_data[1].shape:
            return
        self.tmp_data[0] += data[0]
        self.tmp_data[1] = data[1]
        tmp_time = datetime.now().timestamp()
        if tmp_time - self.last_plot_time > self.fleshPeriod:
            self.plot_current_frame()
            self.last_plot_time = tmp_time

    def plot_current_frame(self, text=None):
        try:
            if self.freq is None or len(self.tmp_data) < 2:
                return
            data = copy.deepcopy(self.tmp_data)
            center_freq = self.center_freq_input.value()

            # --- multi-thread UI: 取当前选择 ---
            tid = self.thread_spin.value()
            chv = self.channel_spin.value()

            # 2) 如果切到某个 thread，则校验 channel 合法性（不存在就弹窗）
            if tid != -1 and self.thread_channels:
                rng = self._valid_channel_range_for_thread(tid)
                if rng is None:
                    # 不存在的 thread（理论上 on_thread_changed 已兜底，这里再次保护）
                    self.alert(title="Warning",
                               text=f"Thread {tid} 不存在于数据中。")
                    return
                ch0, ch1 = rng
                if chv != -1 and not (ch0 <= chv <= ch1):
                    self.alert(title="Warning",
                               text=f"Thread {tid} 的通道范围为 0..{ch1}，当前选择 {chv} 非法。")
                    return

            # 3) 标题文案（仅改变标题，不改变数据选择逻辑；数据仍按你现有通道拼接）
            self.title_suffix = None
            if tid != -1 and chv != -1:
                # thread-{thread_id}:ch-{channel_id}
                self.title_suffix = f"thread-{tid}:ch-{chv}"
            elif tid != -1 and chv == -1:
                # thread-{thread_id}:ch-{ch_first} ==> ch-{ch_end}
                ch0, ch1 = self._overall_channel_bounds(tid)
                self.title_suffix = f"thread-{tid}:ch-{ch0} ==> ch-{ch1}"
            elif tid == -1 and chv == -1:
                # thread-{t_first}ch-{ch_first} ==>  thread-{t_end}ch-{ch_end}
                tf, tl = self._overall_thread_bounds()
                # 若没有多线程信息，用 0..channels-1 填充
                ch0, ch1 = self._overall_channel_bounds(None)
                if tf is None or tl is None:
                    self.title_suffix = f"thread-0ch-{ch0} ==> thread-0ch-{ch1}"
                else:
                    # 末尾通道用最后一个 thread 的范围末端
                    _, last_ch1 = self._overall_channel_bounds(tl)
                    self.title_suffix = f"thread-{tf}ch-{ch0} ==> thread-{tl}ch-{last_ch1}"
            # 其他组合（tid == -1 且 chv != -1）未在需求中定义，保持默认标题

            ichan = self.channel_spin.value()
            if ichan == -1:
                freq = np.concatenate(self.freq, axis=0) + center_freq
                amp = np.concatenate(data[0])
                # amp = vdiflib.power_to_db(data[0])
                phase = np.concatenate(data[1])
            else:
                freq = self.freq[ichan] + center_freq
                amp = data[0][ichan]
                # amp = vdiflib.power_to_db(data[0])
                phase = data[1][ichan]

            peak_max = self.reduce_spin.value()

            # 找到振幅中最大的 n 个索引
            if peak_max > 0:
                filter_amp = amp < peak_max
                self.amp_line.set_data(freq[filter_amp], amp[filter_amp])
                self.phase_line.set_data(freq[filter_amp], phase[filter_amp])
            else:
                self.amp_line.set_data(freq, amp)
                self.phase_line.set_data(freq, phase)

            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            self.canvas.draw()
        except Exception as e:
            self.ax1.clear()
            self.ax2.clear()
            if text:
                self.ax1.text(0.5, 0.5, text, ha='center', transform=self.ax1.transAxes)
            else:
                self.ax1.text(0.5, 0.5, f"Error: {e}", ha='center', transform=self.ax1.transAxes)
            self.canvas.draw()

    def set_frame(self, text=None):
        # if not self.tmp_data:
        #     return
        self.figure.clear()
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)

        # 统一字体大小
        label_fontsize = 12
        title_fontsize = 14
        tick_fontsize = 10

        # 设置坐标轴标签及字体大小
        self.ax1.set_xlabel("Frequency (MHz)", fontsize=label_fontsize)
        self.ax1.set_ylabel("Amplitude", fontsize=label_fontsize)
        self.ax2.set_xlabel("Frequency (MHz)", fontsize=label_fontsize)
        self.ax2.set_ylabel("Phase (radians)", fontsize=label_fontsize)

        # 振幅图
        # ax1.plot(freq, np.abs(spectrum), color='tab:blue', label='Amplitude')
        amp_title = "Amplitude Spectrum"
        phs_title = "Phase Spectrum"
        if self.title_suffix:
            amp_title += f" [{self.title_suffix}]"
            phs_title += f" [{self.title_suffix}]"
        self.ax1.set_title(amp_title, fontsize=14, fontweight='bold')
        self.ax2.set_title(phs_title, fontsize=14, fontweight='bold')
        self.ax1.set_title("Amplitude Spectrum", fontsize=title_fontsize, fontweight='bold')
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        # self.ax1.legend(fontsize=tick_fontsize)
        self.ax1.tick_params(axis='both', labelsize=tick_fontsize)
        self.ax1.margins(x=0)  # 紧贴x轴范围

        # 相位图
        # ax2.plot(freq, np.angle(spectrum), color='tab:orange', label='Phase')
        self.ax2.set_title("Phase Spectrum", fontsize=title_fontsize, fontweight='bold')
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        # self.ax2.legend(fontsize=tick_fontsize)
        self.ax2.tick_params(axis='both', labelsize=tick_fontsize)
        self.ax2.margins(x=0)
        
        self.amp_line = self.ax1.plot([], [], color='tab:blue')[0]
        self.phase_line = self.ax2.plot([], [], color='tab:orange')[0]

        # 调整子图间距，避免标签重叠
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

class PlotUpdateThread(threading.Thread):
    def __init__(self, data_queue: queue.Queue, ui_object):
        super().__init__()
        self.data_queue = data_queue
        self.ui_object = ui_object
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            # try:
            # 阻塞式获取数据，超时可设置为None无限等待
            data = self.data_queue.get(block=True)
            if data is None:
                break
            # print("PlotUpdateThread get data:", data.shape)
            # 将数据填充到界面对象（假设有一个update_data方法）
            if hasattr(self.ui_object, 'update_data'):
                self.ui_object.update_data(np.array(data))
            # 通知队列任务完成（可选）
            # self.data_queue.task_done()
            # except Exception as e:
            #     e.with_traceback()
            #     # 捕获异常防止线程退出，可适当打印日志
            #     print(f"PlotUpdateThread error: {e}")

    def stop(self):
        self._stop_event.set()
        # 如果线程在阻塞队列获取，发送一个None或特殊标志唤醒线程退出
        self.data_queue.put(None)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = VDIFViewer()
    # setup stylesheet
    #apply_stylesheet(app, theme='light_blue.xml')
    #qdarktheme.setup_theme()
    viewer.show()
    sys.exit(app.exec_())
