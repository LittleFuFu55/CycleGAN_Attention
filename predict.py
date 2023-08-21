#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from utils import *
from model import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa
import tensorflow_datasets as tfds

tfds.disable_progress_bar()
autotune = tf.data.AUTOTUNE


# In[2]:


def main():
    parser = argparse.ArgumentParser(description='Predict CycleGAN')
    parser.add_argument('--predict_image_path', type=str, default='./predict_image', help='Path to prdictive image saving')
    parser.add_argument('--test_G_folder_path', type=str, default='./banana/test/Class C/', help='Test main_transfer domain')
    parser.add_argument('--weights_path', type=str, default='./model_checkpoints_banana/cyclegan_checkpoints.100', help='Path to checkpoint')
    parser.add_argument('--predict_num', type=int,default=6, help='Number of predictive images')
  
    parser.add_argument('--raw_image_size', type=int, default=286, help='raw input image size')
    parser.add_argument('--train_image_size', type=int, default=256, help='train image size')
    parser.add_argument('--buffer_size', type=int, default=256, help='dataset shuffle image size')
    parser.add_argument('--batch_size', type=int, default=1, help='Input batch size')

    parser.add_argument('--ngf', type=int, default=64, help='number of base filters in generator')
    parser.add_argument('--ndbg', type=int, default=2, help='number of down sampling block in generator')
    parser.add_argument('--nrb', type=int, default=9, help='number of residual block in generator')
    parser.add_argument('--nub', type=int, default=2, help='number of up sampling blcok in generator')
    parser.add_argument('--ndf', type=int, default=64, help='number of base filters in discriminator')
    parser.add_argument('--ndbd', type=int, default=3, help='number of base filters in discriminator')

    args = parser.parse_args()

    # #Weights initializer for the layers.
    # kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    # #Gamma initializer for instance normalization.
    # gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    test_G_folder_path=args.test_G_folder_path
    test_G_file_list=os.listdir(test_G_folder_path)

    orig_img_size = (args.raw_image_size, args.raw_image_size)
    # Size of the random crops to be used during training.
    input_img_size = (args.train_image_size, args.train_image_size, 3)

    def preprocess_train_image(img, label):
        # Random flip
        img = tf.image.random_flip_left_right(img)
        # Resize to the original size first
        img = tf.image.resize(img, [*orig_img_size])
        # Random crop to 256X256
        img = tf.image.random_crop(img, size=[*input_img_size])
        # Normalize the pixel values in the range [-1, 1]
        img = normalize_img(img)
        return img


    def preprocess_test_image(img, label):
        # Only resizing and normalization for the test images.
        img = tf.image.resize(img, [input_img_size[0], input_img_size[1]])
        img = normalize_img(img)
        return img


    def get_image_test_g(size=args.buffer_size): 
        for file in test_G_file_list:
            img = cv2.imread(test_G_folder_path + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(size,size),interpolation=cv2.INTER_AREA)
            labs=np.array([0], dtype=np.int32).reshape(-1)
            yield img,labs # Your supposed to yield in a loop

    test_C = tf.data.Dataset.from_generator(get_image_test_g,output_shapes=((args.buffer_size,args.buffer_size,3), (1,)),output_types=(tf.uint8,tf.int64))

    # Apply the preprocessing operations to the test data
    test_C = (
        test_C.map(preprocess_test_image, num_parallel_calls=autotune)
        .cache()
        .shuffle(args.buffer_size)
        .batch(args.batch_size))

    # Get the generators
    gen_G = get_resnet_generator(filters=args.ngf,input_img_size =(args.train_image_size,args.train_image_size,3),num_downsampling_blocks=args.ndbg,num_residual_blocks=args.nrb,num_upsample_blocks=args.nub,name="generator_G")
    gen_F = get_resnet_generator(filters=args.ngf,input_img_size =(args.train_image_size,args.train_image_size,3),num_downsampling_blocks=args.ndbg,num_residual_blocks=args.nrb,num_upsample_blocks=args.nub,name="generator_F")
    # Get the discriminators
    disc_X = get_discriminator(filters=args.ndf,input_img_size =(args.train_image_size,args.train_image_size,3), num_downsampling= args.ndbd,name="discriminator_X")
    disc_Y = get_discriminator(filters=args.ndf,input_img_size =(args.train_image_size,args.train_image_size,3), num_downsampling= args.ndbd,name="discriminator_Y")

    class CycleGan(keras.Model):
        def __init__(
            self,
            generator_G,
            generator_F,
            discriminator_X,
            discriminator_Y,
        ):
            super().__init__()
            self.gen_G = generator_G
            self.gen_F = generator_F
            self.disc_X = discriminator_X
            self.disc_Y = discriminator_Y

    cycle_gan_model = CycleGan(
        generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
    )

    weight_file = args.weights_path
    cycle_gan_model.load_weights(weight_file).expect_partial()
    print("Weights loaded successfully")

    _, ax = plt.subplots(args.predict_num, 2, figsize=(10, 15))
    for i, img in enumerate(test_C.take(args.predict_num)):
        prediction = cycle_gan_model.gen_G(img, training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

        ax[i, 0].imshow(img)
        ax[i, 1].imshow(prediction)
        ax[i, 0].set_title("Input image")
        ax[i, 0].set_title("Input image")
        ax[i, 1].set_title("Translated image")
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")

        prediction = keras.utils.array_to_img(prediction)
        prediction.save("{}/predicted_img_{i}.png".format(args.predict_image_path,i=i))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()



