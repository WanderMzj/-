import sys
from PySide6.QtWidgets import QApplication
from gui import ImageApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ImageApp()
    # 默认最大化
    win.showMaximized()
    sys.exit(app.exec())
