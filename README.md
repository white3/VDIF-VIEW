# VDIF Viewer

一个面向使用者的 VDIF 频谱查看工具。  
支持读取 `.vdif` 文件，显示幅度谱和相位谱，并在右侧展示关键 VDIF 头字段。

## 功能概览

- 打开 VDIF 文件并自动解析头信息
- 自动推测常用处理参数并填入设置框
- 支持按 `channel` 查看单通道或全部通道频谱
- 支持按 `thread` 切换（用于多线程数据的浏览）
- 可调整 FFT 点数、刷新频率、振幅显示上限

## 环境要求

- Python 3.9+（建议）
- Windows / Linux / macOS（GUI 依赖 PyQt5）

> 目前实际使用环境：Python 3.9.7 / Windows 11

依赖见 `requirements.txt`：

- `pyqt5`
- `pyqtdarktheme`
- `matplotlib`
- `numpy`
- `baseband`

## 安装与启动

在项目根目录执行：

```bash
pip install -r requirements.txt
python main.py
```

## 使用流程（快速上手）

1. 点击 `Select VDIF File` 选择 `.vdif` 文件。
2. 程序会先解析头信息并显示在右侧表格。
3. 若自动推测成功，`VDIF Settings` 会自动填入类似 `8000-512-16-2`。
4. 点击 `Plot` 开始处理与绘图。
5. 可通过 `Cur Chan` / `Cur Thread` 查看不同通道与线程的数据。

## 主要界面控件说明

- `VDIF Settings`：处理参数字符串，格式为  
  `<VDIF Body Length bytes>-<bps MHz>-<num channels>-<bits per sample>`  
  示例：`8000-512-16-2`
- `Enable Settings`：勾选后可手动修改 `VDIF Settings`
- `FFT Size`：每次 FFT 的点数（越大频率分辨率越高，但速度更慢）
- `Cur Chan`：当前通道；`-1` 表示显示全部通道拼接结果
- `Cur Thread`：当前线程；`-1` 表示全部线程
- `RR`：刷新周期（秒）
- `View Max Amp`：振幅上限过滤（小于等于 `-1` 表示不过滤）

## 自动推测参数是怎么来的

选择文件后，程序会尝试从 VDIF 头中推测配置：

- `VDIF Body Length`：`DATA_FRAME_LENGTH - 32`
- `bps (MHz)`：根据第一秒内帧数与每帧 payload 大小估算
- `num channels / bits per sample`：直接读取头字段
- `CHANNELS_BAND_MHz`：按采样率关系估算每通道带宽（用于展示和排查）

推测流程（简述）：

1. 从文件开头顺序读帧头；
2. 找到第一处 `SECONDS_FROM_EPOCH` 变化点（秒边界）；
3. 用该秒内最后一帧的 `DATA_FRAME_NUMBER + 1` 作为每秒帧数；
4. 结合 `DATA_FRAME_LENGTH`、`NUM_CHANNELS`、`BITS_PER_SAMPLE` 推算速率与带宽。

## 什么时候要手动填写 `VDIF Settings`

如果出现以下情况，建议手动填写：

- 文件长度不足 1 秒
- 时间字段不连续或异常
- 自动推测值明显不符合采集配置

手动格式：

`<VDIF Body Length bytes>-<bps MHz>-<num channels>-<bits per sample>`

例如：`8000-512-16-2`

## 当前适配范围与注意事项

- 默认按 **32-byte VDIF header** 解析与跳转
- `NUM_CHANNELS` 需为 2 的幂（解析有约束）
- 支持 `DATA_TYPE=real/complex`
- 优先针对 **2-bit** 量化数据进行解码
- 多线程数据可在界面中切换查看，但请确认你的数据组织方式与当前实现一致

## 常见问题

### 1) 点击 `Plot` 后没有图或报错

先检查：

- `VDIF Settings` 是否正确
- `FFT Size` 是否过大导致处理慢
- 数据文件是否完整、是否包含有效帧

### 2) 自动推测失败

这是正常保护机制。请勾选 `Enable Settings` 后手动填写配置。

### 3) 频谱看起来不对

通常与参数不匹配有关（尤其是 `bps / channels / bits`）。  
建议先用已知正确参数手动输入进行对照。

## 相关文档

- 头解析模块说明：`docs/vdifheader.md`