# ========================================================================
# Resize, Pad Image to Square Shape and Keep Its Aspect Ratio With Python
# ========================================================================

import os
import argparse
import sys
import random

from PIL import Image, ImageFile
import shutil


import tensorflow as tf

FLAGS = None

MAX_PER_CLASSES = 5
NUM_CLASSES = 10


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    cls_lst = os.listdir(FLAGS.original_dir)
    for i, cls in enumerate(cls_lst):
        # if i >= NUM_CLASSES:
        #     break

        cls_path = os.path.join(FLAGS.original_dir, cls)
        img_lst = os.listdir(cls_path)
        random.shuffle(img_lst)

        for idx, img in enumerate(img_lst):
            if idx >= MAX_PER_CLASSES:
                break

            old_img_path = os.path.join(cls_path, img)
            new_img_path = os.path.join(FLAGS.target_dir, img)
            shutil.copyfile(old_img_path, new_img_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--original_dir',
        type=str,
        default='/home/ace19/dl_data/materials/validation',
        help='Where is image to load.')
    parser.add_argument(
        '--target_dir',
        type=str,
        default='/home/ace19/dl_data/materials/query',
        help='Where is resized image to save.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
