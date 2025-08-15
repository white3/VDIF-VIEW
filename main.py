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

class VDIFViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VDIF Viewer - With Stats Sidebar")
        self.resize(1480, 800)
        self.vdif_path = None
        # data cache for background processing
        self.vdifqueue = queue.Queue()
        self.tmp_data = []
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
        self.init_ui()

    def closeEvent(self, event):
        print("Closing VDIF Viewer")
        self.plotthread.stop()
        if self.prcthread:
            self.prcthread.stopProcess()
        return super().closeEvent(event)

    def init_ui(self):
        self.setWindowIcon(QIcon('./icon.png'))
        main_layout = QHBoxLayout()
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
        self.thread_spin.valueChanged.connect(self.plot_current_frame)
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
        self.tableview.setFixedWidth(400)
        self.tableview.setColumnWidth(0, 240)
        self.tableview.setColumnWidth(1, 160)

        # 关联 QTableView控件和 Model
        self.tableview.setModel(self.model)
        right_layout.addWidget(self.tableview)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)
        self.start_button.setEnabled(False)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 
            "Open VDIF File", "", "VDIF Files (*.vdif);;All Files (*)")
        if path:
            self.file_label.setText(f"Selected: {path}")
            self.vdif_path = path
            self.stats = vdiflib.analyze_vdif_file(path)
            self.display_stats(self.stats)
            self.ready2plot = True
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

    def alert(self, title="Error", text="The VDIF data duration should exceed 1 second!"):
        dlg = QMessageBox(self)
        dlg.setWindowTitle(title)
        dlg.setText(text)
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.setIcon(QMessageBox.Critical)
        dlg.exec()

    def start_background_processing(self):
        self.start_button.setEnabled(False)
        if self.prcthread:
            self.prcthread.stats["running"] = False
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
        self.tmp_data[0] += data[0]
        self.tmp_data[1] = data[1]
        tmp_time = datetime.now().timestamp()
        if tmp_time - self.last_plot_time > self.fleshPeriod:
            self.plot_current_frame()
            self.last_plot_time = tmp_time

    def plot_current_frame(self, text=None):
        try:
            data = copy.deepcopy(self.tmp_data)

            ichan = self.channel_spin.value()
            if ichan == -1:
                freq = np.concatenate(self.freq, axis=0) + center_freq
                amp = np.concatenate(data[0])
                # amp = vdiflib.power_to_db(data[0])
                phase = np.concatenate(data[1])
            else:
                center_freq = self.center_freq_input.value()
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
