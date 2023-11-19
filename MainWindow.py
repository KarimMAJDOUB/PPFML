from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.uic import loadUi
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        loadUi("./PPFML.ui", self)
        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec())
