# 项目部署文档

本文档基于以下 `requirements.txt`，帮助你快速在新环境中搭建并运行本 PySide6+OpenCV 图像处理应用。

```text
PySide6>=6.5
opencv-python>=4.7
numpy>=1.24
matplotlib>=3.6
```

---

## 一、系统与软件要求

1. 操作系统  
   - Windows 10/11、macOS 10.15+、常见 Linux 发行版（Ubuntu、Fedora、Debian 等）  
2. Python  
   - 版本：3.8 ~ 3.11  
   - 建议最新版补丁，如 3.10.12、3.11.6  
3. 硬件  
   - 内存：至少 4 GB，推荐 8 GB  
   - 磁盘：≥200 MB 可用空间

---


## 二、安装依赖

1. 将以下内容保存为 `requirements.txt` 放在项目根目录：  
   ```text
   PySide6>=6.5
   opencv-python>=4.7
   numpy>=1.24
   matplotlib>=3.6
   ```
2. 在激活的虚拟环境中执行：  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. 安装完成后，可通过 `pip list` 或 `pip show` 确认版本

---

## 三、项目目录结构

```
ImageApp/
├── main.py             # 入口脚本
├── gui.py              # 界面与交互逻辑
├── image_utils.py      # 图像处理算法封装
└── requirements.txt    # 依赖列表
```

---

## 四、运行项目

1. 确保虚拟环境已激活  
2. 在项目根目录执行：  
   ```bash
   python main.py
   ```
3. 程序窗口将自动最大化，使用「文件→打开」导入图片，即刻可见并开始交互式处理

---

## 五、功能概览

- 菜单栏  
  - **文件**：打开/保存  
  - **操作**：裁剪、水平翻转、垂直翻转、撤销、复原  
  - **分析**：灰度直方图、彩色直方图、BGR/HSV 通道展示  
  - **滤波**：平滑/锐化 参数对话  
- 右侧面板  
  - 滑块：亮度、对比度、色调、宽度、⾼度（锁定宽高比）  
  - 下拉：插值方式  
- 裁剪：拖拽矩形选区、确认后裁剪  
- 滤波：三级联动（类型→域→算子）+ 系数 0–100%

---
