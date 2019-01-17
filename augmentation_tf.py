import os
import cv2
import numpy as np
import tensorflow as tf

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    image = tf.convert_to_tensor(image, tf.float32)
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                # pass
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    image = tf.convert_to_tensor(image, tf.float32)
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=bbox,
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return cropped_image, distort_bbox


if __name__ == "__main__":
    img = cv2.imread('./test/img/v7_00001.jpg')
    h, w, c = img.shape
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_norm = img / 255.0
    img_pre = distort_color(img_norm, fast_mode=False, color_ordering=3)
    with tf.Session() as sess:
        img = img_pre.eval() * 255
        # cv2.imshow('dwa', img)
        cv2.imwrite('test_3.jpg',img)

        with open('./test/label/v7_00001.txt', 'r') as rf:
            lines = rf.readlines()
            bboxes = []
            for line in lines:
                bbox = []
                info = line.split('\t')
                # print(info)
                if info[9] == '2\n':
                    continue
                # print(info[9])
                ymin = min(int(info[2]), int(info[4]), int(info[6]), int(info[8])) / float(h)
                ymax = max(int(info[2]), int(info[4]), int(info[6]), int(info[8])) / float(h)
                xmin = min(int(info[1]), int(info[3]), int(info[5]), int(info[7])) / float(w)
                xmax = max(int(info[1]), int(info[3]), int(info[5]), int(info[7])) / float(w)
                #
                # bbox.append(int(info[1]))
                # bbox.append(int(info[2]))
                # bbox.append(int(info[3]))
                # bbox.append(int(info[4]))
                # bbox.append(int(info[5]))
                # bbox.append(int(info[6]))
                # bbox.append(int(info[7]))
                # bbox.append(int(info[8]))
                # bbox.append()
                bboxes.append([ymin, xmin, ymax, xmax])
                # cv2.polylines(img, [np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]], [bbox[4], bbox[5]],
                #                               [bbox[6], bbox[7]]])], True, (255, 0, 0), 2)
        # print(np.array([bboxes]).shape)
        cropped_image, distort_bbox = distorted_bounding_box_crop(img_norm, np.array([bboxes]))
        ci = cropped_image.eval() * 255
        h, w, c = ci.shape

        print(distort_bbox.eval())
        for box in distort_bbox.eval()[0]:
            cv2.rectangle(ci, (int(box[1]*w), int(box[0]*h)), (int(box[3]*w), int(box[2]*h)), (255, 0, 0), 2)

        cv2.imwrite('bbox.jpg', ci)
