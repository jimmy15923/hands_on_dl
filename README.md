# 快速上手深度學習實務
## 課前準備: 安裝所需套件
請參考[此網址](http://tensorflowkeras.blogspot.tw/2017/08/tensorflowkeraswindows_29.html?m=1) 完成以下所有安裝步驟:
```
下載/安裝 Anaconda (已包含 python 及許多套件) → 建立虛擬環境 → 安裝 tensorflow → 安裝 keras → 執行範例程式
```

0. [Anaconda 下載網址](https://www.anaconda.com/download/)，建議下載 3.6 版本
1. 打開 CMD，以下 code 建立虛擬環境
```
conda create --name myenv python=3.5 anaconda
```
2. 建立完成後，輸入 acitvate myenv 啟動虛擬環境，並按照順序安裝 tensorflow 及 keras
```
pip install tensorflow 
pip install keras
```
3. 下載 https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py 這份範例程式並執行測試。
```python
python mnist_mlp.py
```
+ 若能成功執行，表示安裝成功。


+ 由於作業系統環境因人而異，若有任何安裝上的問題，請不吝來信詢問: jimmy15923@iis.sinica.edu.tw。先謝謝大家的海涵，希望各位能在 9/27 上課前完成安裝流程，謝謝！

## 下載程式、投影片及資料
資料：
執行以下的 code，會自動下載 CIFAR10 的資料
```python
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```
程式：
TBA
投影片：
TBA

## (optional) 安裝運算加速庫
1. 下載並執行 https://github.com/tw-cmchang/hand-on-dl/blob/master/checkblas.py 測試是否有安裝運算加速庫。
2. 安裝 Anaconda 的學員，也確認是否有通過 checkblas.py 測試。儘管 Anaconda 內有 mkl 運算加速庫。
3. 安裝 openblas，請依 https://github.com/tw-cmchang/hand-on-dl/blob/master/openblas_installation.pdf 。
4. 另外，若需要安裝 openblas 可至 https://drive.google.com/drive/folders/0ByfnsehogjWtbndTY3JncE95bjQ 下載 (謝謝 Chih-Fan)。

## (optional) GPU 安裝 (train CNN 必須有 GPU 才能跑的快，需要新型的 NVIDIA 顯示卡，至少 GTX 650)
若要使用 GPU 來訓練模型，必須完成以下步驟
1. 安裝 gpu 版本的 tensorflow
2. 安裝 CUDA
3. 安裝 cuDNN

### 如之前的教學，但是原先安裝的 tensorflow 是 CPU 版本，需進到虛擬環境中，安裝 tensorflow GPU 版本
```
pip install tensorflow-gpu
```

### 在 Windows 10 安裝 CUDA & cuDNN 可以參考下列網址
1. [安裝 CUDA](http://ankivil.com/installing-keras-theano-and-dependencies-on-windows-10/)
2. [安裝 cuDNN](http://ankivil.com/making-theano-faster-with-cudnn-and-cnmem-on-windows-10/)

### 在 ubuntu 上安裝可以參考下列影片
* https://www.youtube.com/watch?v=wjByPfSFkBo

### 沒有 GPU 的折衷方案 (Windows 10, openBLAS CPU 加速)
* 請安裝 [openBLAS](https://github.com/chihfanhsu/dnn_hand_by_hand/blob/master/openblas_install.pdf)

## Other Questions
+ 有學員回報在 win10 安裝 Anaconda2 後，使用 pip install theano/ pip install keras 出現下方錯誤訊息：
```pyhon
UnicodeDecodeError: 'ascii' codec can't decode byte 0xb8 in position 0: ordinal not in range(128)
```
此為 Anaconda2 的預設編碼問題。請在 Anaconda2\Lib\site-packages 裡增加一個 sitecustomize.py，內容如下：
```python
import sys 
sys.setdefaultencoding('gbk')
```
之後在 pip install theano/keras 試試看，若有問題請再來信。謝謝該位熱心的同學提供解法 :)。
