# gui.py
import cv2, numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QFileDialog,
    QSlider, QSpinBox, QPushButton,
    QHBoxLayout, QVBoxLayout, QGroupBox,
    QMessageBox, QRubberBand, QDialog,
    QFormLayout, QComboBox, QDialogButtonBox, QCheckBox
)
from PySide6.QtCore import Qt, QRect, QPoint, QSize
from PySide6.QtGui import QImage, QPixmap,QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import QScrollArea
import image_utils as iu

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.rubberBand.setStyleSheet("border:3px solid yellow; background:rgba(255,255,0,50);")
        self.origin = QPoint()
        self.crop_callback = None

    def mousePressEvent(self, e):
        if e.button()==Qt.LeftButton and self.crop_callback:
            self.origin = e.pos()
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()

    def mouseMoveEvent(self, e):
        if self.rubberBand.isVisible():
            self.rubberBand.setGeometry(QRect(self.origin, e.pos()).normalized())

    def mouseReleaseEvent(self, e):
        if e.button()==Qt.LeftButton and self.crop_callback:
            rect = self.rubberBand.geometry()
            self.rubberBand.hide()
            self.crop_callback(rect)

class FilterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("滤波参数设置")
        form = QFormLayout(self)

        # 一级：锐化/平滑
        self.combobox1 = QComboBox(); self.combobox1.addItems(['锐化','平滑'])
        form.addRow("类型：", self.combobox1)
        # 二级：空域/频域
        self.combobox2 = QComboBox(); form.addRow("模式：", self.combobox2)
        # 三级：算子
        self.combobox3 = QComboBox(); form.addRow("算子：", self.combobox3)
        # 系数：0–100%
        self.kslider = QSlider(Qt.Horizontal); self.kslider.setRange(0,100); self.kslider.setValue(50)
        self.kspin   = QSpinBox(); self.kspin.setRange(0,100); self.kspin.setValue(50)
        self.kslider.valueChanged .connect(self.kspin.setValue)
        self.kspin.valueChanged   .connect(self.kslider.setValue)
        h = QHBoxLayout(); h.addWidget(self.kslider); h.addWidget(self.kspin)
        form.addRow("强度：", h)
        # 按钮
        btns = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addWidget(btns)
        # 动态联动
        self.combobox1.currentIndexChanged .connect(self.update_combo2_options)
        self.combobox2.currentIndexChanged .connect(self.update_combo3_options)
        self.update_combo2_options()

    def update_combo2_options(self):
        self.combobox2.clear()
        if self.combobox1.currentText() == '锐化':
            self.combobox2.addItems(['空域锐化','频域锐化'])
        else:
            self.combobox2.addItems(['空域平滑','频域平滑'])
        self.update_combo3_options()

    def update_combo3_options(self):
        self.combobox3.clear()
        mode = self.combobox2.currentText()
        if mode == '空域锐化':
            self.combobox3.addItems(['roberts算子','sobel算子','laplacian算子','LoG算子','prewitt算子'])
        elif mode == '空域平滑':
            self.combobox3.addItems(['均值滤波','统计滤波'])
        elif mode == '频域锐化':
            self.combobox3.addItems(['理想高通滤波','巴特沃斯高通滤波','高斯高通滤波'])
        else:
            self.combobox3.addItems(['理想低通滤波','巴特沃斯低通滤波','高斯低通滤波'])

    def get_params(self):
        # 修正了属性名
        return {
            'type':     self.combobox1.currentText(),
            'mode':     self.combobox2.currentText(),
            'operator': self.combobox3.currentText(),
            'k':        self.kspin.value()
        }

class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("全能图像处理")
        self._orig = None; self._proc = None; self.history = []

        # 菜单
        mb = self.menuBar()
        fm = mb.addMenu("文件")
        fm.addAction("打开",  self.open_image)
        fm.addAction("保存",  self.save_image)
        om = mb.addMenu("操作")
        om.addAction("裁剪",  self.start_crop)
        om.addAction("水平翻转", lambda:self.apply_op('flipH'))
        om.addAction("垂直翻转", lambda:self.apply_op('flipV'))
        om.addAction("撤销",  self.undo)
        
        am = mb.addMenu("分析")
        am.addAction("绘制灰度直方图", lambda:self.show_hist('gray'))
        am.addAction("绘制彩色直方图", lambda:self.show_hist('color'))
        am.addAction("展示BGR图层",     lambda:self.show_channels('BGR'))
        am.addAction("展示HSV图层",     lambda:self.show_channels('HSV'))
        fm2 = mb.addMenu("滤波")
        fm2.addAction("平滑/锐化", self.filter_dialog)
        dm = mb.addMenu("检测")
        dm.addAction("边缘/直线检测", self.edge_line_dialog)


        # 中心 & 裁剪回调（改为 QScrollArea 包裹）
        w = QWidget(); self.setCentralWidget(w)
        hbox = QHBoxLayout(w)
        self.img_label = ImageLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.crop_callback = self.do_crop
        # 用 QScrollArea 包裹 QLabel
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.img_label)
        self.scroll_area.setWidgetResizable(True)
        hbox.addWidget(self.scroll_area, 3)

        # 右侧面板：滑块 + 缩放
        ctrl = QVBoxLayout(); hbox.addLayout(ctrl, 1)
        self.sliders = {}
        for key, rng, title in [
            ('brightness',(-100,100),'亮度'),
            ('contrast',  (-100,100),'对比度'),
            ('hue',       (-90,90),  '色调'),
        ]:
            gb = QGroupBox(title)
            v  = QVBoxLayout(gb)
            hl = QHBoxLayout()
            s  = QSlider(Qt.Horizontal); s.setRange(*rng); s.setValue(0)
            sp = QSpinBox();     sp.setRange(*rng); sp.setValue(0)
            s.valueChanged.connect(sp.setValue); sp.valueChanged.connect(s.setValue)
            s.valueChanged.connect(self.apply_sliders)
            hl.addWidget(s); hl.addWidget(sp)
            v.addLayout(hl)
            ctrl.addWidget(gb)
            self.sliders[key] = s

        # —— 缩放：**两个滑块** + 锁比 + 插值 —— #
        gb2=QGroupBox("缩放设置"); vb=QVBoxLayout(gb2)

        vb.addWidget(QLabel("宽度"))
        self.slider_w=QSlider(Qt.Horizontal); vb.addWidget(self.slider_w)
        vb.addWidget(QLabel("高度"))
        self.slider_h=QSlider(Qt.Horizontal); vb.addWidget(self.slider_h)

        self.chk_ratio=QCheckBox("锁定宽高比"); vb.addWidget(self.chk_ratio)

        vb.addWidget(QLabel("插值方式"))
        self.combo_interp=QComboBox(); self.combo_interp.addItems(list(iu.INTERP_MAP.keys()))
        vb.addWidget(self.combo_interp)

        ctrl.addWidget(gb2)

        # Connect 缩放滑块
        self.slider_w.valueChanged.connect(self.on_resize_slider)
        self.slider_h.valueChanged.connect(self.on_resize_slider)
        self.chk_ratio.stateChanged.connect(self.on_resize_slider)
        self.combo_interp.currentIndexChanged.connect(self.on_resize_slider)

        ctrl.addStretch()

    def push_history(self):
        if self._proc is not None:
            self.history.append(self._proc.copy())
            if len(self.history)>20: self.history.pop(0)

    # gui.py 中把 update_display 改成：
    def update_display(self):
        img = self._proc
        h, w = img.shape[:2]
        bytes_line = 3*w
        rgb = img[..., ::-1].copy()
        qimg = QImage(rgb.data, w, h, bytes_line, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        # 不再做 .scaled(label.size()), 而是让 label 自身变成图片大小
        self.img_label.setPixmap(pix)
        self.img_label.resize(pix.size())


     # —— 打开图片立刻显示 & 初始化缩放滑块 —— #
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "*.png *.jpg *.bmp")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            return

        # 1. 保存原图，清空历史
        self._orig = img.copy()
        self.history.clear()

        # 2. 计算视口大小
        vw = self.scroll_area.viewport().width()
        vh = self.scroll_area.viewport().height()
        orig_h, orig_w = img.shape[:2]

        # 3. 适配视口的初始缩放比例（不放大，只缩小）
        fit_ratio = min(vw/orig_w, vh/orig_h, 1.0)
        init_w    = max(1, int(orig_w * fit_ratio))
        init_h    = max(1, int(orig_h * fit_ratio))

        # 4. 设置放缩滑块范围为原图的 [1,1.5×]，上限调低
        max_w = int(orig_w * 1.5)
        max_h = int(orig_h * 1.5)
        self.slider_w.setRange(1, max_w)
        self.slider_h.setRange(1, max_h)

        # 5. 初始值阻断信号后设置为适配尺寸
        self.slider_w.blockSignals(True)
        self.slider_h.blockSignals(True)
        self.slider_w.setValue(init_w)
        self.slider_h.setValue(init_h)
        self.slider_w.blockSignals(False)
        self.slider_h.blockSignals(False)

        # 6. 通过 on_resize_slider 生成并显示首次缩放结果
        self._proc = self._orig.copy()
        self.on_resize_slider()


         # —— 复原功能 —— #
    def reset_all(self):
        if self._orig is None: return
        self._proc=self._orig.copy(); self.history.clear()
        # 重置所有滑块
        for s in self.sliders.values():
            s.blockSignals(True); s.setValue(0); s.blockSignals(False)
        # 重置缩放滑块
        h,w=self._orig.shape[:2]
        for sl,val in [(self.slider_w,w),(self.slider_h,h)]:
            sl.blockSignals(True); sl.setValue(val); sl.blockSignals(False)
        self.chk_ratio.setChecked(False)
        self.combo_interp.setCurrentIndex(1)  # 默认双线性
        self.update_display()

        # —— 缩放联动 & 锁比 —— #
    def on_resize_slider(self, *_):
        if self._orig is None: return
        self.push_history()
        # 1. 先取滑块值并 clamp
        w = max(1, self.slider_w.value())
        h = max(1, self.slider_h.value())
        if self.chk_ratio.isChecked():
            oh,ow=self._orig.shape[:2]; ar=ow/oh
            sender=self.sender()
            if sender==self.slider_w:
                h=int(w/ar)
                self.slider_h.blockSignals(True); self.slider_h.setValue(h); self.slider_h.blockSignals(False)
            else:
                w=int(h*ar)
                self.slider_w.blockSignals(True); self.slider_w.setValue(w); self.slider_w.blockSignals(False)
        self._proc = iu.resize_interpolation(self._orig, w, h, self.combo_interp.currentText())
        self.update_display()
        
    def save_image(self):
        if self._proc is None: return
        path,_ = QFileDialog.getSaveFileName(self, "保存图片", "", "*.png *.jpg")
        if not path: return
        cv2.imwrite(path, self._proc)

    def undo(self):
        if not self.history: return
        self._proc = self.history.pop()
        self.update_display()

    def apply_sliders(self):
        if self._orig is None: return
        img = self._orig.copy()
        b = self.sliders['brightness'].value()
        c = self.sliders['contrast'].value()
        h = self.sliders['hue'].value()
        img = iu.adjust_brightness_contrast(img, b, c)
        img = iu.adjust_hue(img, h)
        self._proc = img
        self.update_display()

    def apply_resize(self):
        if self._proc is None: return
        self.push_history()
        w = self.spin_w.value(); h = self.spin_h.value()
        if self.chk_ratio.isChecked():
            oh, ow = self._orig.shape[:2]
            r = ow/oh
            h = int(w / r)
            self.spin_h.setValue(h)
        self._proc = iu.resize_interpolation(self._proc, w, h, self.combo_interp.currentText())
        self.update_display()

    def start_crop(self):
        self.statusBar().showMessage("请在图片上拖拽选区后确认")

    def do_crop(self, rect: QRect):
        if self._proc is None: return
        if QMessageBox.question(
            self, "确认裁剪", "是否裁剪选定区域？",
            QMessageBox.Yes | QMessageBox.No
        ) != QMessageBox.Yes:
            self.statusBar().clearMessage()
            return
        self.push_history()
        # 坐标映射
        lw, lh = self.img_label.width(), self.img_label.height()
        ih, iw = self._proc.shape[:2]
        ratio = min(lw/iw, lh/ih)
        dx = (lw - iw*ratio)/2; dy = (lh - ih*ratio)/2
        x0 = int((rect.x()-dx)/ratio); y0 = int((rect.y()-dy)/ratio)
        w  = int(rect.width()/ratio); h  = int(rect.height()/ratio)
        x0, y0 = max(0,x0), max(0,y0)
        crop = iu.crop_img(self._proc, (x0,y0,w,h))
        if crop.size>0:
            self._proc = crop
            self.update_display()
        self.statusBar().clearMessage()

    def apply_op(self, op):
        if self._proc is None: return
        self.push_history()
        if op=='flipH':
            self._proc = iu.flip_img(self._proc, horizontal=True)
        else:
            self._proc = iu.flip_img(self._proc, vertical=True)
        self.update_display()

    def filter_dialog(self):
        if self._proc is None: return
        dlg = FilterDialog(self)
        if dlg.exec() != QDialog.Accepted: return
        p = dlg.get_params()
        self.push_history()
        self._proc = iu.apply_filter(
            self._proc, p['type'], p['mode'], p['operator'], p['k']
        )
        self.update_display()

    def show_hist(self, mode):
        if self._proc is None: return
        fig = iu.plot_gray_hist(self._proc) if mode=='gray' else iu.plot_color_hist(self._proc)
        self._show_fig(fig, '灰度直方图' if mode=='gray' else '彩色直方图')

    def show_channels(self, cs):
        if self._proc is None: return
        fig = iu.plot_channels(self._proc, cs)
        self._show_fig(fig, f'{cs} 通道展示')

    def _show_fig(self, fig, title):
        dlg = QDialog(self); dlg.setWindowTitle(title)
        canvas = FigureCanvas(fig)
        lay = QVBoxLayout(dlg); lay.addWidget(canvas)
        dlg.resize(600,400); dlg.exec()

    def edge_line_dialog(self):
        if self._proc is None:
            return

        # 1. 创建对话框
        dlg = QDialog(self)
        dlg.setWindowTitle("边缘检测与线条检测")
        dlg.resize(800, 600)             # 对话框尺寸
        form = QFormLayout(dlg)

        # 2. 功能下拉
        combo1 = QComboBox()
        combo1.addItems(["边缘检测", "线条变化检测"])
        form.addRow("功能：", combo1)

        # 3. 算子下拉
        combo2 = QComboBox()
        form.addRow("算子：", combo2)

        # 4. 预览区域
        preview_label = QLabel()
        preview_label.setFixedSize(600, 400)
        preview_label.setAlignment(Qt.AlignCenter)
        form.addRow("预览：", preview_label)

        # 5. 确认按钮
        btn = QPushButton("确认")
        form.addRow(btn)

        # 6. 参数映射
        para = {
            "roberts算子":"roberts","sobel算子":"sobel","laplacian算子":"laplacian",
            "LoG算子":"log",       "canny算子":"canny", "prewitt算子":"prewitt",
            "曲线检测":"curve",    "直线检测":"line"
        }

        raw = self._proc.copy()  # 缓存当前图

        # 7. 动态更新算子列表
        def update_ops():
            combo2.clear()
            if combo1.currentText() == "边缘检测":
                combo2.addItems(
                    ["roberts算子","sobel算子","laplacian算子",
                     "LoG算子","canny算子","prewitt算子"]
                )
            else:
                combo2.addItems(["曲线检测","直线检测"])
            update_preview()

        # 8. 生成并显示预览
        def update_preview():
            op1 = combo1.currentText()
            key = para.get(combo2.currentText(), "")
            if not key:
                return

            # 1. 生成预览图 pre（BGR）
            if op1 == "边缘检测":
                pre = iu.edgedetect(raw, key)
            else:
                pre = iu.houghlines(raw, key)

            # 2. BGR→RGB，并强制 C 连续
            # 方法 A：用 cvtColor（会返回连续数组）
            rgb = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
            # 或 方法 B：切片后 copy
            # rgb = pre[..., ::-1].copy()

            # 3. 构造 QImage
            h, w = rgb.shape[:2]
            bpl  = 3 * w
            qimg = QImage(rgb.data, w, h, bpl, QImage.Format_RGB888)

            # 4. 缩放并显示
            pix = QPixmap.fromImage(qimg).scaled(
                preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation)
            preview_label.setPixmap(pix)


        # 9. 确认后把结果写回主图
        def on_confirm():
            op1 = combo1.currentText()
            key = para.get(combo2.currentText(), "")
            self.push_history()
            if op1 == "边缘检测":
                self._proc = iu.edgedetect(raw, key)
            else:
                self._proc = iu.houghlines(raw, key)
            self.update_display()
            dlg.accept()

        # 10. 绑定信号
        combo1.currentIndexChanged.connect(update_ops)
        combo2.currentIndexChanged.connect(update_preview)
        btn.clicked.connect(on_confirm)

        # 11. 初始化并执行对话
        update_ops()
        dlg.exec()
