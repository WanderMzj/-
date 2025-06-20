# image_utils.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# —— Matplotlib 强制中文显示 —— #
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# —— 插值方式映射 —— #
INTERP_MAP = {
    '最近邻插值': cv2.INTER_NEAREST,
    '双线性插值': cv2.INTER_LINEAR,
    '双三次插值': cv2.INTER_CUBIC,
    'LANCZOS':      cv2.INTER_LANCZOS4,
}

# —— 亮度 / 对比度 —— #
def adjust_brightness_contrast(img, brightness=0, contrast=0):
    """
    brightness: 偏移量 [-100,100]
    contrast: 对比度增量 [-100,100]
    """
    alpha = 1.0 + (contrast / 100.0)
    beta  = brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# —— 色调 (Hue) 调节 —— #
def adjust_hue(img, hue_shift=0):
    """
    hue_shift: 色相偏移 [-90,90]
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(int) + hue_shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# —— 缩放（指定宽高 + 插值） —— #
def resize_interpolation(img, width, height, interp_name):
    """
    width, height: 目标尺寸
    interp_name: INTERP_MAP 中的键
    """
    interp = INTERP_MAP.get(interp_name, cv2.INTER_LINEAR)
    return cv2.resize(img, (width, height), interpolation=interp)

# —— 旋转 —— #
def rotate_img(img, angle=0):
    """
    angle: 旋转角度（度），正数逆时针
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

# —— 翻转 —— #
def flip_img(img, horizontal=False, vertical=False):
    """
    horizontal=True 水平翻转
    vertical=True   垂直翻转
    """
    if horizontal and vertical:
        return cv2.flip(img, -1)
    if horizontal:
        return cv2.flip(img, 1)
    if vertical:
        return cv2.flip(img, 0)
    return img

# —— 裁剪 —— #
def crop_img(img, rect):
    """
    rect: (x, y, w, h)
    """
    x, y, w, h = rect
    return img[y:y+h, x:x+w]

# —— 空间域平滑 —— #
def spatial_smooth(img, operator, k):
    """
    operator: '均值滤波' or '统计滤波'
    k: 卷积核尺寸（整数）
    """
    k = max(1, int(k))
    if operator == '均值滤波':
        return cv2.blur(img, (k, k))
    else:
        k = k if k % 2 == 1 else k + 1
        return cv2.medianBlur(img, k)

# 空域锐化（增强系数 k/50+1，k=100 时因子=3）
def spatial_sharpen(img, operator, k):
    fimg = img.astype(np.float32)
    if operator == 'roberts算子':
        kernel = np.array([[1,0],[0,-1]], np.float32)
        res = cv2.filter2D(fimg, -1, kernel)
    elif operator == 'sobel算子':
        gx = cv2.Sobel(fimg, cv2.CV_32F, 1,0,ksize=3)
        gy = cv2.Sobel(fimg, cv2.CV_32F, 0,1,ksize=3)
        res = np.sqrt(gx*gx+gy*gy)
    elif operator == 'laplacian算子':
        res = cv2.Laplacian(fimg, cv2.CV_32F)
    elif operator == 'LoG算子':
        blur= cv2.GaussianBlur(fimg, (0,0), sigmaX=1)
        res = cv2.Laplacian(blur, cv2.CV_32F)
    elif operator == 'prewitt算子':
        px = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],np.float32)
        py = px.T
        gx = cv2.filter2D(fimg, -1, px)
        gy = cv2.filter2D(fimg, -1, py)
        res = np.sqrt(gx*gx+gy*gy)
    else:
        return img
    factor = 1 + k/50.0
    out = cv2.convertScaleAbs(res * factor)
    return out

# —— 频域高/低通滤波示例 —— #
def freq_filter(img, operator, k):
    h, w = img.shape[:2]
    crow, ccol = h//2, w//2
    # D0 根据 k 百分比决定
    D0 = max(h, w) * (k / 100.0)
    # 构造掩码
    mask = np.zeros((h, w), np.float32)
    if '低通' in operator:
        cv2.circle(mask, (ccol, crow), int(D0), 1, -1)
    else:
        cv2.circle(mask, (ccol, crow), int(D0), 0, -1)
        mask = 1 - mask
    # 分通道处理
    channels = cv2.split(img)
    out = []
    for ch in channels:
        dft = cv2.dft(np.float32(ch), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        # 应用掩码
        dft_shift[...,0] *= mask
        dft_shift[...,1] *= mask
        idft = cv2.idft(np.fft.ifftshift(dft_shift))
        mag  = cv2.magnitude(idft[:,:,0], idft[:,:,1])
        out.append(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    return cv2.merge(out)

# —— 统一调用滤波 —— #
def apply_filter(img, filter_type, domain, operator, k):
    """
    filter_type: '平滑' or '锐化'
    domain: '空间域' or '频域'
    operator: 算子名称
    k: 强度 0–100
    """
    if filter_type == '平滑':
        if domain == '空间域':
            return spatial_smooth(img, operator, k)
        else:
            return freq_filter(img, operator, k)
    else:
        if domain == '空间域':
            return spatial_sharpen(img, operator, k)
        else:
            return freq_filter(img, operator, k)

# —— 灰度直方图 —— #
def plot_gray_hist(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fig, ax = plt.subplots()
    ax.hist(gray.ravel(), bins=256, color='black')
    ax.set_title('灰度直方图')
    ax.set_xlabel('灰度值')
    ax.set_ylabel('像素数')
    return fig

# —— 彩色直方图 —— #
def plot_color_hist(img):
    fig, ax = plt.subplots()
    colors = ('b','g','r')
    names  = ('蓝','绿','红')
    for i, col in enumerate(colors):
        ax.hist(img[:,:,i].ravel(), bins=256, color=col, alpha=0.5)
    ax.set_title('彩色直方图')
    ax.set_xlabel('像素值')
    ax.set_ylabel('像素数')
    ax.legend(names)
    return fig

# —— 通道展示 BGR/HSV —— #
def plot_channels(img, cs='BGR'):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    if cs == 'BGR':
        titles = ['蓝','绿','红']
        chans  = [0,1,2]
        data   = img
    else:
        hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        titles = ['色相','饱和度','明度']
        chans  = [0,1,2]
        data   = hsv
    for ax, idx in zip(axes, chans):
        ax.imshow(data[:,:,idx], cmap='gray')
        ax.set_title(titles[idx])
        ax.axis('off')
    fig.suptitle(f'{cs} 通道展示')
    return fig
