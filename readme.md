针对目标检测的数据增强脚本，目前支持如下方法：

1.图像增强（不会改变bbox）

    1.1.增加椒盐噪音
    1.2.转为灰度
    1.3.RBG->BGR
    1.4.调整亮度
    1.5.色度增强
    1.6.锐度增强
    
2.图像坐标增强（改变bbox）

    2.1.剪裁
    2.2.平移
    2.3.镜像
    2.4.翻转
    2.5.任意角度旋转