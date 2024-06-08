import sys
import subprocess

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout

app = QApplication(sys.argv)

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Game")

        taoButton = QPushButton()
        taoButton.setCheckable(True)
        taoButton.clicked.connect(self.run_taogame)
      
        budButton = QPushButton() 
        budButton.setCheckable(True)
        budButton.clicked.connect(self.run_budgame)

        taoButton.setStyleSheet("""
            QPushButton {
                border-radius: 15px;
                padding: 10px;
                height: 800px;
                background-image: url(resources/taoism/taoism.jpeg);
                background-position: center; 
            }
            QPushButton:pressed {
                border: 2px solid #8A2BE2;
            }
        """)

        budButton.setStyleSheet("""
            QPushButton {
                border-radius: 15px;
                padding: 10px;
                height: 800px;
                background-image: url(resources/buddhism/buddhism.jpeg);
                background-position: center; 
                background-size: contain;
            }
            QPushButton:pressed {
                border: 2px solid #8A2BE2;
            }
        """)

        self.setFixedSize(QSize(1200, 800))
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)

        # Create a horizontal layout
        layout = QHBoxLayout()
        layout.addWidget(taoButton)
        layout.addWidget(budButton)
        self.setLayout(layout)

    def run_taogame(self):
        subprocess.run(["python", "taogame.py"])
    
    def run_budgame(self):
        subprocess.run(["python", "budgame.py"])

window = MainWindow()
window.show()  # Windows are hidden by default.

app.exec()
