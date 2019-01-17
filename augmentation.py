import os
from PIL import Image, ImageEnhance
# DataAugmentForObjectDetection
"""
针对目标检测的数据增强脚本，目前支持如下方法：
1.图像增强（不会改变bbox）
    1.1.增加椒盐噪音
    1.2.转为灰度
    1.3.RBG->BGR
    1.4.调整亮度
    1.5.色度增强
    1.6.锐度增强
"""

class ImageEnhance():

    def random_brightness(self, img, brightness):
        """
        亮度增强
        :param img:
        :param brightness:
        :return:
        """
        enh_bri = ImageEnhance.Brightness(img)
        image_brightened = enh_bri.enhance(brightness)
        return np.array(image_brightened)

    def enhance_color(self, img, color):
        """
        色度增强
        :param img:
        :param color:
        :return:
        """
        img = Image.fromarray(img)
        enh_col = ImageEnhance.Color(img)
        image_colored = enh_col.enhance(color)
        return np.array(image_colored)

    def enhance_contrast(self, img, contrast=5):
        """
        对比度增强
        :param img:
        :param contrast:
        :return:
        """
        img = Image.fromarray(img)
        enh_con = ImageEnhance.Contrast(img)
        image_contrasted = enh_con.enhance(contrast)
        return np.array(image_contrasted)

    def enhance_sharpness(self, img, sharpness):
        """
        锐度增强
        :param img:
        :param sharpness:
        :return:
        """
        img = Image.fromarray(img)
        enh_sha = ImageEnhance.Sharpness(img)
        image_sharped = enh_sha.enhance(sharpness)
        return np.array(image_sharped)
