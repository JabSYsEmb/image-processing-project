#!./noise_project/bin/python

import sys
import os
import numpy as np
import cv2 as cv

from filters import *

from PyQt5 import QtGui, uic
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QListWidget, QListWidgetItem,\
    QVBoxLayout, QButtonGroup, QSlider, QFileDialog, QPushButton, QTabWidget

def cv2qim(image):
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    q_image = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
    return q_image

class NoiseRestorationAPP(QMainWindow):
    noise_signal = pyqtSignal(int, int, int)
    restored_signal = pyqtSignal(int, int, int)
    compress_signal = pyqtSignal(int, int)

    def __init__(self):
        super(NoiseRestorationAPP, self).__init__()
        uic.loadUi('APP.ui', self)

        # main variables
        self.img_width = int(app.primaryScreen().size().width()/2 - 200)
        self.img_height = int(app.primaryScreen().size().height()/2 - 40)
        self.original = None
        self.noisy = None

        # defining widgets
        self.images_list = self.findChild(QListWidget, 'images_list')
        self.original_label = self.findChild(QLabel, 'original_label')
        self.noisy_label = self.findChild(QLabel, 'noisy_label')
        self.restored_label = self.findChild(QLabel, 'restored_label')
        self.images_layout = self.findChild(QVBoxLayout, 'images_layout')
        self.noise_radio_group = self.findChild(QButtonGroup, 'noise_radio_group')
        self.filter_radio_group = self.findChild(QButtonGroup, 'filter_radio_group')
        self.noiseProp1_label = self.findChild(QLabel, 'noiseProp1_label')
        self.noiseProp2_label = self.findChild(QLabel, 'noiseProp2_label')
        self.noiseProp1_slider = self.findChild(QSlider, 'noiseProp1_slider')
        self.noiseProp2_slider = self.findChild(QSlider, 'noiseProp2_slider')
        self.prop1Value_label = self.findChild(QLabel, 'prop1Value_label')
        self.prop2Value_label = self.findChild(QLabel, 'prop2Value_label')
        self.filterProp1_label = self.findChild(QLabel, 'filterProp1_label')
        self.filterProp2_label = self.findChild(QLabel, 'filterProp2_label')
        self.filterProp3_label = self.findChild(QLabel, 'filterProp3_label')
        self.filterPropDisp1_label = self.findChild(QLabel, 'filterPropDisp1_label')
        self.filterPropDisp2_label = self.findChild(QLabel, 'filterPropDisp2_label')
        self.filterPropDisp3_label = self.findChild(QLabel, 'filterPropDisp3_label')
        self.filterProp1_slider = self.findChild(QSlider, 'filterProp1_slider')
        self.filterProp2_slider = self.findChild(QSlider, 'filterProp2_slider')
        self.filterProp3_slider = self.findChild(QSlider, 'filterProp3_slider')
        self.browse_button = self.findChild(QPushButton, 'browse_button')
        self.tabWidget = self.findChild(QTabWidget, 'tabWidget')
        self.compressed_label = self.findChild(QLabel, "compressed_label")
        self.quant_label = self.findChild(QLabel, "quant_label")
        self.block_label = self.findChild(QLabel, "block_label")
        self.quant_slider = self.findChild(QSlider, 'quant_slider')
        self.block_slider = self.findChild(QSlider, 'block_slider')
        self.noisyTitle_label = self.findChild(QLabel, 'noisyTitle_label')
        self.restoredTitle_label = self.findChild(QLabel, 'restoredTitle_label')

        # connecting signals
        self.images_list.itemClicked.connect(self.update_original)
        self.noise_radio_group.idClicked.connect(self.update_noise_gui)
        self.filter_radio_group.idClicked.connect(self.update_restored_gui)
        self.noise_signal.connect(self.update_noisy)
        self.restored_signal.connect(self.update_restored)
        self.noiseProp1_slider.valueChanged.connect(self.noise_slider_moved)
        self.noiseProp2_slider.valueChanged.connect(self.noise_slider_moved)
        self.filterProp1_slider.valueChanged.connect(self.restored_slider_moved)
        self.filterProp2_slider.valueChanged.connect(self.restored_slider_moved)
        self.filterProp3_slider.valueChanged.connect(self.restored_slider_moved)
        self.browse_button.clicked.connect(self.load)
        self.tabWidget.currentChanged.connect(self.change_tab_index_event)
        self.quant_slider.valueChanged.connect(self.compress)
        self.block_slider.valueChanged.connect(self.compress)

        # initial functions
        self.populate_list("./images/")

    def change_tab_index_event(self, index):
        if index == 1:
            self.noisy_label.hide()
            self.restored_label.hide()
            self.compressed_label.show()
            self.noisyTitle_label.hide()
            self.restoredTitle_label.hide()
            self.compress()
        elif index == 0:
            self.noisy_label.show()
            self.restored_label.show()
            self.compressed_label.hide()
            self.original_label.adjustSize()
            self.noisy_label.adjustSize()
            self.restored_label.adjustSize()
            self.noisyTitle_label.show()
            self.restoredTitle_label.show()
            self.showNormal()
            self.showMaximized()

    def compress(self):
        item = self.images_list.currentItem()
        image = item.data(Qt.UserRole)
        quantization = int(self.quant_label.text())
        block = int(self.block_label.text())

        compressed = jpeg_comp(image, quantization, block)

        q_image = cv2qim(compressed)
        if q_image.height() > self.img_height:
            q_image = q_image.scaledToHeight(self.img_height)

        self.compressed_label.setPixmap(QPixmap(q_image))

    def update_original(self):
        item = self.images_list.currentItem()

        image = item.data(Qt.UserRole)
        q_image = cv2qim(image)
        if q_image.height() > self.img_height:
            q_image = q_image.scaledToHeight(self.img_height)

        self.original_label.setPixmap(QPixmap(q_image))
        self.noise_slider_moved()
        if self.tabWidget.currentIndex() == 1:
            self.compress()

    def noise_slider_moved(self):
        filter = self.noise_radio_group.checkedId()
        prop1 = self.noiseProp1_slider.sliderPosition()
        prop2 = self.noiseProp2_slider.sliderPosition()
        self.noise_signal.emit(filter, prop1, prop2)

    def update_noise_gui(self, button):

        # gaussian
        if button == -2:
            self.noiseProp2_label.show()
            self.prop2Value_label.show()
            self.noiseProp2_slider.show()

            self.noiseProp1_label.setText("Mean =")
            self.noiseProp1_slider.setRange(0, 10)
            self.noiseProp1_slider.setTickInterval(1)

            self.noiseProp2_label.setText("Variance =")
            self.noiseProp2_slider.setRange(0, 100)
            self.noiseProp2_slider.setTickInterval(1)

            prop1 = self.noiseProp1_slider.sliderPosition()
            prop2 = self.noiseProp2_slider.sliderPosition()
            self.noise_signal.emit(-2, prop1, prop2)

        # gamma
        elif button == -3:
            self.noiseProp2_label.show()
            self.prop2Value_label.show()
            self.noiseProp2_slider.show()

            self.noiseProp1_label.setText("Mean =")
            self.noiseProp1_slider.setRange(0, 100)
            self.noiseProp1_slider.setTickInterval(1)

            self.noiseProp2_label.setText("Variance =")
            self.noiseProp2_slider.setRange(0, 100)
            self.noiseProp2_slider.setTickInterval(1)

            prop1 = self.noiseProp1_slider.sliderPosition()
            prop2 = self.noiseProp2_slider.sliderPosition()
            self.noise_signal.emit(-3, prop1, prop2)

        # Salt and pepper
        elif button == -4:
            self.noiseProp2_label.hide()
            self.noiseProp2_slider.hide()
            self.prop2Value_label.hide()

            self.noiseProp1_label.setText("Probability =")
            self.noiseProp1_slider.setRange(0, 10)
            self.noiseProp1_slider.setTickInterval(1)

            prop1 = self.noiseProp1_slider.sliderPosition()
            prop2 = self.noiseProp2_slider.sliderPosition()
            self.noise_signal.emit(-4, prop1, prop2)

        # Exponential
        elif button == -5:
            self.noiseProp2_label.hide()
            self.prop2Value_label.hide()
            self.noiseProp2_slider.hide()

            self.noiseProp1_label.setText("Mean =")
            self.noiseProp1_slider.setRange(0, 100)
            self.noiseProp1_slider.setTickInterval(1)

            prop1 = self.noiseProp1_slider.sliderPosition()
            prop2 = self.noiseProp2_slider.sliderPosition()
            self.noise_signal.emit(-5, prop1, prop2)

        # Rayleigh
        elif button == -6:
            self.noiseProp2_label.hide()
            self.noiseProp2_slider.hide()
            self.prop2Value_label.hide()

            self.noiseProp1_label.setText("Mean =")
            self.noiseProp1_slider.setRange(0, 100)
            self.noiseProp1_slider.setTickInterval(1)

            prop1 = self.noiseProp1_slider.sliderPosition()
            prop2 = self.noiseProp2_slider.sliderPosition()
            self.noise_signal.emit(-6, prop1, prop2)

        # uniform
        elif button == -7:
            self.noiseProp2_label.show()
            self.noiseProp2_slider.show()
            self.prop1Value_label.show()
            self.prop2Value_label.show()

            self.noiseProp1_label.setText("a =")
            self.noiseProp1_slider.setRange(0, 255)
            self.noiseProp1_slider.setTickInterval(1)

            self.noiseProp2_label.setText("b =")
            self.noiseProp2_slider.setRange(0, 255)
            self.noiseProp2_slider.setTickInterval(1)

            prop1 = self.noiseProp1_slider.sliderPosition()
            prop2 = self.noiseProp2_slider.sliderPosition()
            self.noise_signal.emit(-2, prop1, prop2)

    def update_noisy(self, noise_filter, prop1, prop2):
        original = self.images_list.currentItem().data(Qt.UserRole)

        if noise_filter == -2:
            noisy = cv.add(original,noise_gaussian(original.shape, prop1, prop2))
            

        elif noise_filter == -3:
            prop1 /= 10
            noisy = cv.add(original, noise_gamma(original.shape, prop1, prop2))

        elif noise_filter == -4:
            prop1 /= 100
            noisy = original * noise_salt_pepper(original.shape, prop1)
            noisy.astype(np.uint8)

        elif noise_filter == -5:
            noisy = cv.add(original, noise_exponential(original.shape, prop1))

        elif noise_filter == -6:
            noisy = cv.add(original, noise_rayleigh(original.shape, prop1))

        elif noise_filter == -7:
            if prop2 < prop2:
                prop2 = prop1
            noisy = cv.add(original, noise_uniform(original.shape, prop1, prop2))

        else:
            noisy = self.original
            print("couldn't specify filter type")

        self.prop1Value_label.setText(str(prop1))
        self.prop2Value_label.setText(str(prop2))

        q_image = cv2qim(noisy)
        if q_image.width() > self.img_width:
            q_image = q_image.scaledToWidth(self.img_width)
        if q_image.height() > self.img_height:
            q_image = q_image.scaledToHeight(self.img_height)

        self.noisy_label.setPixmap(QPixmap(q_image))
        self.noisy = noisy
        self.restored_slider_moved()

    def restored_slider_moved(self):
        prop1 = self.filterProp1_slider.sliderPosition()
        prop2 = self.filterProp2_slider.sliderPosition()
        prop3 = self.filterProp3_slider.sliderPosition()
        self.restored_signal.emit(prop1, prop2, prop3)

    def update_restored_gui(self, button):
        # Median
        if button == -3:
            self.filterPropDisp1_label.setText("K size")
            self.filterProp1_slider.setRange(1, 50)

            self.filterPropDisp2_label.hide()
            self.filterProp2_label.hide()
            self.filterProp2_slider.hide()

            self.filterPropDisp3_label.hide()
            self.filterProp3_label.hide()
            self.filterProp3_slider.hide()

            self.restored_slider_moved()

        # Gaussian
        elif button == -2:
            self.filterPropDisp1_label.setText("Radius =")
            self.filterProp1_slider.setRange(0, 100)
            self.filterProp1_slider.setTickInterval(1)

            self.filterPropDisp2_label.hide()
            self.filterProp2_label.hide()
            self.filterProp2_slider.hide()

            self.filterPropDisp3_label.hide()
            self.filterProp3_label.hide()
            self.filterProp3_slider.hide()

            self.restored_slider_moved()

        # Arithmetic mean
        elif button == -4:
            self.filterPropDisp1_label.setText("K size =")
            self.filterProp1_slider.setRange(1, 2)

            self.filterPropDisp2_label.hide()
            self.filterProp2_label.hide()
            self.filterProp2_slider.hide()

            self.filterPropDisp3_label.hide()
            self.filterProp3_label.hide()
            self.filterProp3_slider.hide()

            self.restored_slider_moved()

        # Bilateral
        elif button == -7:
            self.filterPropDisp1_label.setText("d =")
            self.filterProp1_slider.setRange(1, 20)

            self.filterPropDisp2_label.show()
            self.filterProp2_label.show()
            self.filterProp2_slider.show()
            self.filterPropDisp2_label.setText("Sigma color =")
            self.filterProp2_slider.setRange(1, 20)

            self.filterPropDisp3_label.show()
            self.filterProp3_label.show()
            self.filterProp3_slider.show()
            self.filterPropDisp3_label.setText("Sigma space =")
            self.filterProp3_slider.setRange(1, 20)

            prop1 = self.filterProp1_slider.sliderPosition()
            prop2 = self.filterProp2_slider.sliderPosition()
            prop3 = self.filterProp3_slider.sliderPosition()
            #self.restored_signal.emit(prop1, prop2, prop3)

        # Max
        elif button == -6:
            self.filterPropDisp1_label.setText("K size =")
            self.filterProp1_slider.setRange(1, 5)

            self.filterPropDisp2_label.hide()
            self.filterProp2_label.hide()
            self.filterProp2_slider.hide()

            self.filterPropDisp3_label.hide()
            self.filterProp3_label.hide()
            self.filterProp3_slider.hide()

            self.restored_slider_moved()

        # Min
        elif button == -5:
            self.filterPropDisp1_label.setText("K size =")
            self.filterProp1_slider.setRange(1, 5)

            self.filterPropDisp2_label.hide()
            self.filterProp2_label.hide()
            self.filterProp2_slider.hide()

            self.filterPropDisp3_label.hide()
            self.filterProp3_label.hide()
            self.filterProp3_slider.hide()

            self.restored_slider_moved()

    def update_restored(self, prop1, prop2, prop3):
        filterIndex = self.filter_radio_group.checkedId()

        if filterIndex == -3:
            prop1 = 2 * prop1 + 1
            self.filterProp1_label.setText(str(prop1))
            restored = median_filter(self.noisy, prop1)

        elif filterIndex == -2:
            self.filterProp1_label.setText(str(prop1))
            restored = gaussian_filter(self.noisy, prop1)

        elif filterIndex == -7:
            prop1 *= 5
            prop2 *= 5
            self.filterProp1_label.setText(str(prop1))
            self.filterProp2_label.setText(str(prop2))
            self.filterProp3_label.setText(str(prop3))
            restored = bilateral_filter(self.noisy, prop1, prop2, prop3)

        elif filterIndex == -6:
            prop1 = 2*prop1 + 1
            restored = max_filter(self.noisy, prop1)

        elif filterIndex == -4:
            prop1 = 2 * prop1 + 1
            self.filterProp1_label.setText(str(prop1))
            restored = arithmatic_mean_filter(self.noisy, prop1)

        elif filterIndex == -5:
            prop1 = 2*prop1 + 1
            self.filterProp1_label.setText(str(prop1))
            restored = min_filter(self.noisy, prop1)

        q_image = cv2qim(restored)
        if q_image.width() > self.img_width:
            q_image = q_image.scaledToWidth(self.img_width)
        if q_image.height() > self.img_height:
            q_image = q_image.scaledToHeight(self.img_height)

        self.restored_label.setPixmap(QPixmap(q_image))

    def populate_list(self, folder_path):
        self.images_list.clear()
        images = os.listdir(folder_path)

        for image in images:
            path = os.path.join(folder_path, image)

            icon = QtGui.QIcon()
            icon.addPixmap(QPixmap(path), QtGui.QIcon.Normal, QtGui.QIcon.Off)

            item = QListWidgetItem()
            item.setIcon(icon)
            item.setData(Qt.UserRole, cv.imread(path))
            self.images_list.addItem(item)

        self.images_list.setCurrentItem(self.images_list.item(0))
        self.update_original()
        self.noise_signal.emit(-2, 0, 0)
        self.compressed_label.hide()
        self.showMaximized()

    def load(self):
        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.populate_list(file)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = NoiseRestorationAPP()
    demo.show()
    sys.exit(app.exec_())


