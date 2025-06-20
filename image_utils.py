# image_utils.py
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 用于算子映射
# 键跟下面 gui.py 里 combobox3 对应
# roberts算子→"roberts"，sobel算子→"sobel"…
ls1 = ["roberts算子","sobel算子","laplacian算子","LoG算子","prewitt算子",
       "均值滤波","统计滤波","理想高通滤波","巴特沃斯高通滤波","高斯高通滤波",
       "理想低通滤波","巴特沃斯低通滤波","高斯低通滤波"]
ls2 = ["roberts","sobel","laplacian","log","prewitt",
       "mean","statistical","ideal_high","butter_high","gauss_high",
       "ideal_low","butter_low","gauss_low"]
para = dict(zip(ls1, ls2))

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
def spatialsmooth(img, operator, ksize):
    """空间域平滑：均值 or 统计"""
    k = max(1, int(ksize))
    if operator == "mean":
        return cv2.blur(img, (k, k))
    else:
        k = k if k % 2 == 1 else k+1
        return cv2.medianBlur(img, k)

# 空域锐化（增强系数 k/50+1，k=100 时因子=3）
# ——— 新的空间域锐化 ———
def spatial_sharpen(img, method, k):
    """
    空域锐化：直接把边缘强度加回原图
      img: BGR uint8
      method: "roberts","sobel","laplacian","log","canny","prewitt"
      k: 锐化强度 0–100，对应 sharpness=k/50.0
    """
    # 1. 转成 int16 防止溢出
    bgr16 = img.astype(np.int16)
    # 2. 灰度图（uint8），用于边缘检测
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 计算 uint8 边缘
    if method == "roberts":
        kernel = np.array([[1,0],[0,-1]], np.float32)
        edge  = cv2.filter2D(gray, -1, kernel)
    elif method == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        edge = cv2.convertScaleAbs(cv2.addWeighted(gx,1,gy,1,0))
    elif method == "laplacian":
        edge = cv2.Laplacian(gray, cv2.CV_16S)
        edge = cv2.convertScaleAbs(edge)
    elif method == "log":
        blur = cv2.GaussianBlur(gray, (3,3), sigmaX=0.5)
        edge = cv2.Laplacian(blur, cv2.CV_16S)
        edge = cv2.convertScaleAbs(edge)
    elif method == "prewitt":
        kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float32)
        ky = kx.T
        rx = cv2.filter2D(gray, -1, kx)
        ry = cv2.filter2D(gray, -1, ky)
        edge = cv2.convertScaleAbs(cv2.addWeighted(rx,1,ry,1,0))
    elif method == "canny":
        edge = cv2.Canny(gray, 50, 150)
    else:
        edge = np.zeros_like(gray)

    # 4. 转成 int16 并扩成三通道
    edge16 = edge.astype(np.int16)
    edge3  = np.stack([edge16]*3, axis=2)

    # 5. 叠加到原图
    sharpness = k/50.0                # k=100 -> sharpness=2.0
    res16     = bgr16 + edge3 * sharpness

    # 6. 裁剪并回 uint8
    res = np.clip(res16, 0, 255).astype(np.uint8)
    return res

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
            return spatialsmooth(img, operator, k)
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

def edgedetect(img, operator):
    """
    边缘检测，返回 BGR 格式结果图
    operator: 'roberts','sobel','prewitt','laplacian','log','canny'
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if operator == 'roberts':
        kernel = np.array([[1,0],[0,-1]], np.float32)
        edge = cv2.filter2D(gray, -1, kernel)
    elif operator == 'sobel':
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge = cv2.magnitude(gx, gy)
    elif operator == 'prewitt':
        kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float32)
        ky = kx.T
        gx = cv2.filter2D(gray, -1, kx)
        gy = cv2.filter2D(gray, -1, ky)
        edge = np.hypot(gx, gy)
    elif operator == 'laplacian':
        edge = cv2.Laplacian(gray, cv2.CV_64F)
    elif operator == 'log':
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        edge = cv2.Laplacian(blur, cv2.CV_64F)
    elif operator == 'canny':
        edge = cv2.Canny(gray, 100, 200)
    else:
        edge = np.zeros_like(gray)

    # 归一化并转回 uint8
    edge = cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX)
    edge = edge.astype(np.uint8)
    # 转成 BGR 方便在彩色窗口中显示
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

def houghlines(img, mode):
    """
    线条检测，mode: 'curve' 或 'line'
    返回原图叠加检测结果
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    out   = img.copy()
    if mode == 'line':
        # 概率霍夫直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=30, maxLineGap=10)
        if lines is not None:
            for x1,y1,x2,y2 in lines[:,0]:
                cv2.line(out, (x1,y1), (x2,y2), (0,0,255), 2)
    else:
        # 霍夫圆检测当作“曲线检测”示例
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=10, maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for x,y,r in circles[0]:
                cv2.circle(out, (x,y), r, (0,255,0), 2)
    return out
