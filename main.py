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
from PyQt5.QtGui import (QStandardItemModel, QStandardItem)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import vdiflib
from datetime import datetime, timezone

class VDIFViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VDIF Viewer - With Stats Sidebar")
        self.resize(1200, 800)
        self.current_frame = 0
        self.vdif_path = None
        # data cache for background processing
        self.tmp_data = []
        # data cache for plotting
        self.plotnum = vdiflib.AtomicInt(-1)
        # data cache for processing parameters
        self.stats = {'CHANNELS_BAND': '32.00 MHz', 'INVALID_FLAG': False, 'LEGACY_MODE': False, 'REFERENCE_EPOCH': datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc), 'SECONDS_FROM_EPOCH': 1155000, 'UNASSIGNED_FIELD': 0, 'DATA_FRAME_NUMBER': 1999, 'VDIF_VERSION': 0, 'NUM_CHANNELS': 1, 'DATA_FRAME_LENGTH': 8032, 'BITS_PER_SAMPLE': 2, 'THREAD_ID': 0, 'EXTENDED_DATA_VERSION': 0, 'DATA_TYPE': 'real', 'STATION_ID': 'NA', 'EXTENDED_DATA': {}}
        self.prcthread = None
        self.init_ui()

    def close(self):
        self.prcthread.stats['running'] = False
        self.prcthread.join()
        return super().close()

    def init_ui(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        int_layout = QHBoxLayout()

        self.VDIF_settings_label = QLabel("VDIF Settings:")
        self.VDIF_settings_label.setFixedWidth(75)
        self.VDIF_settings_combo = QLineEdit()
        self.VDIF_settings_combo.setPlaceholderText("512MHz-4ch-2bit")
        self.VDIF_settings_combo.setEnabled(False)
        self.VDIF_settings_combo.setFixedWidth(125)
        self.VDIF_settings_checkbox = QCheckBox()
        self.VDIF_settings_checkbox.setChecked(False)
        self.VDIF_settings_checkbox.toggled.connect(
            lambda checked: self.VDIF_settings_combo.setEnabled(checked)
        )
        self.file_label = QLabel("No file selected")

        file_button = QPushButton("Select VDIF File")
        file_button.clicked.connect(self.select_file)
        file_label_layout = QHBoxLayout()
        file_label_layout.addWidget(self.VDIF_settings_label)
        file_label_layout.addWidget(self.VDIF_settings_combo)
        file_label_layout.addWidget(self.VDIF_settings_checkbox)
        file_label_layout.addWidget(self.file_label)

        self.FFT_size_label = QLabel("FFT Size:")
        self.FFT_size_label.setFixedWidth(75)
        self.FFT_size_spin = QSpinBox()
        self.FFT_size_spin.setMinimum(32)
        self.FFT_size_spin.setMaximum(4096)
        self.FFT_size_spin.setValue(1024)
        int_layout.addWidget(self.FFT_size_label)
        int_layout.addWidget(self.FFT_size_spin)
        int_layout.addWidget(file_button)

        int_layout.addWidget(QLabel("Integration (seconds):"))
        self.integration_spin = QDoubleSpinBox()
        self.integration_spin.setMinimum(0.0)
        self.integration_spin.setValue(0.1)
        int_layout.addWidget(self.integration_spin)
        
        self.start_button = QPushButton("Plot")
        self.start_button.clicked.connect(self.start_background_processing)
        int_layout.addWidget(self.start_button)

        left_layout.addLayout(file_label_layout)
        left_layout.addLayout(int_layout)

        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        self.figure.clear()
        self.plot_current_frame("Please select a VDIF file first.")

        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("< Prev Frame")
        self.next_button = QPushButton("Next Frame >")
        self.prev_button.clicked.connect(self.prev_frame)
        self.next_button.clicked.connect(self.next_frame)

        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.setValue(0)
        self.frame_spin.valueChanged.connect(self.change_current_frame)
        self.frame_spin.setFixedWidth(100)
        self.current_frame_label = QLabel("Current Frame:")
        self.current_frame_label.setFixedWidth(100)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.current_frame_label)
        nav_layout.addWidget(self.frame_spin)
        nav_layout.addWidget(self.next_button)
        left_layout.addLayout(nav_layout)
        left_layout.addWidget(self.canvas)

        self.model = QStandardItemModel(15, 2)
        self.model.setHorizontalHeaderLabels(["VDIFHeaderField", "value"])

        self.tableview = QTableView()
        self.tableview.setAlternatingRowColors(True)
        self.tableview.setFixedWidth(450)
        self.tableview.setColumnWidth(0, 250)
        self.tableview.setColumnWidth(1, 200)

        self.tableview.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 关联 QTableView控件和 Model
        self.tableview.setModel(self.model)
        right_layout.addWidget(self.tableview)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 
            "Open VDIF File", "", "VDIF Files (*.vdif);;All Files (*)")
        if path:
            self.file_label.setText(f"Selected: {path}")
            self.vdif_path = path
            self.stats = vdiflib.analyze_vdif_file(path)
            self.display_stats(self.stats)
            self.ready2plot = True

    def display_stats(self, stats):
        i = 0
        for field_name, field_value in stats.items():
            self.model.setItem(i, 0, QStandardItem(str(field_name)))
            self.model.setItem(i, 1, QStandardItem(str(field_value)))
            i += 1

    def start_background_processing(self):
        self.start_button.setEnabled(True)

        if self.stats.get('CHANNELS_BAND_MHz') is None:
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Error")
            dlg.setText("The VDIF data duration should exceed 1 second!")
            dlg.setStandardButtons(QMessageBox.Ok)
            dlg.setIcon(QMessageBox.Critical)
            dlg.exec()
            return
        if self.prcthread:
            self.prcthread.stats["running"] = False
        self.tmp_data = []
        self.plotnum = vdiflib.AtomicInt(-1)
        self.prcthread = vdiflib.VDIFProcessThread(
            self.integration_spin.value(),
            self.FFT_size_spin.value(),
            self.stats,
            self.vdif_path,
            self.tmp_data,
            self
        )
        if self.ready2plot:
            self.prcthread.setDaemon(False)
            self.prcthread.start()
            self.ready2plot = False

    def plot_current_frame(self):
        if self.current_frame >= self.plotnum.get():
            self.current_frame = self.plotnum.get()
            self.frame_spin.setValue(self.current_frame)
        elif self.current_frame < 0:
            self.current_frame = 0
            self.frame_spin.setValue(self.current_frame)

        try:
            data = np.load(self.tmp_data[self.current_frame])
            spectrum = np.fft.fftshift(np.fft.fft(data))
            freq = np.fft.fftshift(np.fft.fftfreq(len(data)))

            self.amp_line.set_data(freq, np.abs(spectrum))
            self.phase_line.set_data(freq, np.angle(spectrum))

            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            self.canvas.draw()
        except Exception as e:
            self.ax1.clear()
            self.ax2.clear()
            self.ax1.text(0.5, 0.5, f"Error: {e}", ha='center', transform=self.ax1.transAxes)
            self.canvas.draw()

    def set_frame(self, text=None):
        # if not self.tmp_data:
        #     return
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)

        # 统一字体大小
        label_fontsize = 12
        title_fontsize = 14
        tick_fontsize = 10

        # 设置坐标轴标签及字体大小
        ax1.set_xlabel("Frequency (MHz)", fontsize=label_fontsize)
        ax1.set_ylabel("Amplitude", fontsize=label_fontsize)
        ax2.set_xlabel("Frequency (MHz)", fontsize=label_fontsize)
        ax2.set_ylabel("Phase (radians)", fontsize=label_fontsize)

        try:
            data = self.tmp_data[self.current_frame]
            freq = np.array(data[0][1:], dtype=float)
            spectrum = np.array(data[1][1:], dtype=complex)

            # 振幅图
            ax1.plot(freq, np.abs(spectrum), color='tab:blue', label='Amplitude')
            ax1.set_title("Amplitude Spectrum", fontsize=title_fontsize, fontweight='bold')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(fontsize=tick_fontsize)
            ax1.tick_params(axis='both', labelsize=tick_fontsize)
            ax1.margins(x=0)  # 紧贴x轴范围

            # 相位图
            ax2.plot(freq, np.angle(spectrum), color='tab:orange', label='Phase')
            ax2.set_title("Phase Spectrum", fontsize=title_fontsize, fontweight='bold')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(fontsize=tick_fontsize)
            ax2.tick_params(axis='both', labelsize=tick_fontsize)
            ax2.margins(x=0)

        except Exception as e:
            if text:
                ax1.text(0.5, 0.5, text, ha='center', 
                    va='center', fontsize=14, 
                    color='red', fontweight='bold')
            else:
                ax1.text(0.5, 0.5, f"Error: {e}", 
                    ha='center', va='center', 
                    fontsize=14, color='red', 
                    fontweight='bold')
            # ax2.axis('off')

        # 调整子图间距，避免标签重叠
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()
    
    def change_current_frame(self):
        self.current_frame = self.frame_spin.value()
        self.plot_current_frame()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev_frame()
        elif event.key() == Qt.Key_Right:
            self.next_frame()

    def prev_frame(self):
        self.current_frame = max(0, self.current_frame - 1)
        self.frame_spin.setValue(self.current_frame)
        self.plot_current_frame()

    def next_frame(self):
        self.current_frame += 1
        self.frame_spin.setValue(self.current_frame)
        self.plot_current_frame()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = VDIFViewer()
    viewer.show()
    sys.exit(app.exec_())
