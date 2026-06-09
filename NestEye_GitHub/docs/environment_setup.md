### 建立並啟用 Python 虛擬環境

```bash
# 建立虛擬環境
$ python3.10 -m venv yolov7_env

# 啟用虛擬環境
$ source yolov7_env/bin/activate
```
- ==若使用 py 3.12.x 後續安裝 requirements.txt 會z有問題。==

升級 pip 並安裝最新版的 setuptools 和 wheel：
```bash
$ python -m pip install --upgrade pip setuptools wheel
```

___
### Pytorch & CUDA 版本

查看版本，並確認是否卸載原先版本
```bash
# 查看版本
$ pip list

# 卸載原先版本
$ python -m pip uninstall -y torch torchvision torchaudio
```
- ==若 PyTorch 太新，YOLOv7 舊版權重載入方式會不相容==，需要降 PyTorch 版本。

安裝 CUDA 12.1 版本的 PyTorch：
```bash
$ python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

---
### 下載 YOLOv7 原始碼並安裝依賴套件
下載 YOLOv7 GitHub 原始碼並進入 yolov7 資料夾：
```bash
$ git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
```

安裝官方範例所需的 Python 套件：
```bash
$ pip install -r requirements.txt
```

安裝 backports.lzma
- Linux 伺服器的 Python 環境缺少 _lzma 模組。
- torchvision 在匯入資料集工具時會用到 lzma。

```bash
$ pip install backports.lzma
```

下載 YOLOv7‑tiny 預訓練權重：
- 路徑：yolov7/models/yolov7-tiny.pt
```bash
$ wget -O models/yolov7-tiny.pt https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
```