#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


def main():
    parser = argparse.ArgumentParser(description='Train CycleGAN')
    parser.add_argument('--G_folder_path', type=str, default='./banana/train/Class C/', help='main transfer domain')
    parser.add_argument('--F_folder_path', type=str, default='./banana/train/Class D/', help='other transfer domain')
    parser.add_argument('--test_G_folder_path', type=str, default='./banana/test/Class C/', help='test main_transfer domain')
    parser.add_argument('--test_F_folder_path', type=str, default='./banana/test/Class D/', help='test other transfer domain')
    parser.add_argument('--weights_path', type=str, default='./model_checkpoints_banana/', help='path to checkpoint')
    parser.add_argument('--tensorboard_path', type=str,default='./train_logs/banana/', help='location to save tboard records')
  
    parser.add_argument('--raw_image_size', type=int, default=286, help='raw input image size')
    parser.add_argument('--train_image_size', type=int, default=256, help='train image size')
    parser.add_argument('--buffer_size', type=int, default=256, help='dataset shuffle image size')
    parser.add_argument('--batch_size', type=int, default=1, help='Input batch size')
    parser.add_argument('--lr_G_g', type=float, default=0.0002, help='Learning rate for G generator, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta for Adam, default=0.5')
    parser.add_argument('--lr_F_g', type=float, default=0.0002, help='Learning rate for F generator, default=0.0002')
    parser.add_argument('--lr_G_d', type=float, default=0.0002, help='Learning rate for F generator, default=0.0002')
    parser.add_argument('--lr_F_d', type=float, default=0.0002, help='Learning rate for F generator, default=0.0002')
    parser.add_argument('--epochs', type=int,default=10, help='training epochs')
    parser.add_argument('--show_result', type=bool,default=False, help='training temp result show')
    parser.add_argument('--predict_image_save_path', type=str,default='./Fake_image/banana/', help='training temp result save path')
    
    
    
    parser.add_argument('--ngf', type=int, default=64, help='number of base filters in generator')
    parser.add_argument('--ndbg', type=int, default=2, help='number of down sampling block in generator')
    parser.add_argument('--nrb', type=int, default=9, help='number of residual block in generator')
    parser.add_argument('--nub', type=int, default=2, help='number of up sampling blcok in generator')
    parser.add_argument('--ndf', type=int, default=64, help='number of base filters in discriminator')
    parser.add_argument('--ndbd', type=int, default=3, help='number of base filters in discriminator')
     


    args = parser.parse_args()


    # In[ ]:


    # Define the standard image size.
    orig_img_size = (args.raw_image_size, args.raw_image_size)
    # Size of the random crops to be used during training.
    input_img_size = (args.train_image_size, args.train_image_size, 3)
    # Weights initializer for the layers.
    kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    # Gamma initializer for instance normalization.
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    buffer_size = args.buffer_size
    batch_size = args.batch_size

    G_folder_path=args.G_folder_path
    G_file_list=os.listdir(G_folder_path)
    F_folder_path=args.F_folder_path
    F_file_list=os.listdir(F_folder_path)
    
    
    test_G_folder_path=args.test_G_folder_path
    test_G_file_list=os.listdir(test_G_folder_path)
    test_F_folder_path=args.test_F_folder_path
    test_F_file_list=os.listdir(test_F_folder_path)

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

    
    def get_image_g(size=buffer_size):
        for file in G_file_list:
            img = cv2.imread(G_folder_path + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(size,size),interpolation=cv2.INTER_AREA)
            labs=np.array([0], dtype=np.int32).reshape(-1)
            yield img,labs # Your supposed to yield in a loop
        
    def get_image_f(size=buffer_size): 
        for file in F_file_list:
            img = cv2.imread(F_folder_path + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(size,size),interpolation=cv2.INTER_AREA)
            labs=np.array([1], dtype=np.int32).reshape(-1)
            yield img,labs # Your supposed to yield in a loop

    def get_image_test_g(size=buffer_size): 
        for file in test_G_file_list:
            img = cv2.imread(test_G_folder_path + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(size,size),interpolation=cv2.INTER_AREA)
            labs=np.array([0], dtype=np.int32).reshape(-1)
            yield img,labs # Your supposed to yield in a loop

    def get_image_test_f(size=buffer_size):
        for file in test_F_file_list:
            img = cv2.imread(test_F_folder_path + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(size,size),interpolation=cv2.INTER_AREA)
            labs=np.array([1], dtype=np.int32).reshape(-1)
            yield img,labs # Your supposed to yield in a loop

    # In[ ]:
    
    train_C = tf.data.Dataset.from_generator(get_image_g,output_shapes=((buffer_size,buffer_size,3), (1,)),output_types=(tf.uint8,tf.int64))
    train_D = tf.data.Dataset.from_generator(get_image_f,output_shapes=((buffer_size,buffer_size,3), (1,)),output_types=(tf.uint8,tf.int64))
    test_C = tf.data.Dataset.from_generator(get_image_test_g,output_shapes=((buffer_size,buffer_size,3), (1,)),output_types=(tf.uint8,tf.int64))
    test_D = tf.data.Dataset.from_generator(get_image_test_f,output_shapes=((buffer_size,buffer_size,3), (1,)),output_types=(tf.uint8,tf.int64))


    # In[ ]:


    # Apply the preprocessing operations to the training data
    train_C = (
        train_C.map(preprocess_train_image, num_parallel_calls=autotune)
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
    )
    train_D = (
        train_D.map(preprocess_train_image, num_parallel_calls=autotune)
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
    )

    # Apply the preprocessing operations to the test data
    test_C = (
        test_C.map(preprocess_test_image, num_parallel_calls=autotune)
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
    )
    test_D = (
        test_D.map(preprocess_test_image, num_parallel_calls=autotune)
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
    )


    # In[ ]:


    # Get the generators
    gen_G = get_resnet_generator(filters=args.ngf,input_img_size =(args.train_image_size,args.train_image_size,3),num_downsampling_blocks=args.ndbg,num_residual_blocks=args.nrb,num_upsample_blocks=args.nub,gamma_initializer=gamma_init,name="generator_G")
    gen_F = get_resnet_generator(filters=args.ngf,input_img_size =(args.train_image_size,args.train_image_size,3),num_downsampling_blocks=args.ndbg,num_residual_blocks=args.nrb,num_upsample_blocks=args.nub,gamma_initializer=gamma_init,name="generator_F")

    # Get the discriminators
    disc_X = get_discriminator(filters=args.ndf,input_img_size =(args.train_image_size,args.train_image_size,3), num_downsampling= args.ndbd,kernel_initializer=kernel_init,name="discriminator_X")
    disc_Y = get_discriminator(filters=args.ndf,input_img_size =(args.train_image_size,args.train_image_size,3), num_downsampling= args.ndbd,kernel_initializer=kernel_init,name="discriminator_Y")


    # In[ ]:


    class CycleGan(keras.Model):
        def __init__(
            self,
            generator_G,
            generator_F,
            discriminator_X,
            discriminator_Y,
            lambda_cycle=10.0,
            lambda_identity=0.5,
        ):
            super().__init__()
            self.gen_G = generator_G
            self.gen_F = generator_F
            self.disc_X = discriminator_X
            self.disc_Y = discriminator_Y
            self.lambda_cycle = lambda_cycle
            self.lambda_identity = lambda_identity

        def compile(
            self,
            gen_G_optimizer,
            gen_F_optimizer,
            disc_X_optimizer,
            disc_Y_optimizer,
            gen_loss_fn,
            disc_loss_fn,
        ):
            super().compile()
            self.gen_G_optimizer = gen_G_optimizer
            self.gen_F_optimizer = gen_F_optimizer
            self.disc_X_optimizer = disc_X_optimizer
            self.disc_Y_optimizer = disc_Y_optimizer
            self.generator_loss_fn = gen_loss_fn
            self.discriminator_loss_fn = disc_loss_fn
            self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
            self.identity_loss_fn = keras.losses.MeanAbsoluteError()

        def train_step(self, batch_data):
            
            real_x, real_y = batch_data

            with tf.GradientTape(persistent=True) as tape:
                # x2y
                fake_y = self.gen_G(real_x, training=True)
                # y2x
                fake_x = self.gen_F(real_y, training=True)

                # Cycle  x -> y -> x
                cycled_x = self.gen_F(fake_y, training=True)
                # Cycle  y -> x -> y
                cycled_y = self.gen_G(fake_x, training=True)

                # Identity mapping
                same_x = self.gen_F(real_x, training=True)
                same_y = self.gen_G(real_y, training=True)

                # Discriminator output
                disc_real_x = self.disc_X(real_x, training=True)
                disc_fake_x = self.disc_X(fake_x, training=True)

                disc_real_y = self.disc_Y(real_y, training=True)
                disc_fake_y = self.disc_Y(fake_y, training=True)

                # Generator adversarial loss
                gen_G_loss = self.generator_loss_fn(disc_fake_y)
                gen_F_loss = self.generator_loss_fn(disc_fake_x)

                # Generator cycle loss
                cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
                cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

                # Generator identity loss
                id_loss_G = (
                    self.identity_loss_fn(real_y, same_y)
                    * self.lambda_cycle
                    * self.lambda_identity
                )
                id_loss_F = (
                    self.identity_loss_fn(real_x, same_x)
                    * self.lambda_cycle
                    * self.lambda_identity
                )

                # Total generator loss
                total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
                total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

                # Discriminator loss
                disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
                disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

            # Get the gradients for the generators
            grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
            grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

            # Get the gradients for the discriminators
            disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
            disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

            # Update the weights of the generators
            self.gen_G_optimizer.apply_gradients(
                zip(grads_G, self.gen_G.trainable_variables)
            )
            self.gen_F_optimizer.apply_gradients(
                zip(grads_F, self.gen_F.trainable_variables)
            )

            # Update the weights of the discriminators
            self.disc_X_optimizer.apply_gradients(
                zip(disc_X_grads, self.disc_X.trainable_variables)
            )
            self.disc_Y_optimizer.apply_gradients(
                zip(disc_Y_grads, self.disc_Y.trainable_variables)
            )

            return {
                "G_loss": total_loss_G,
                "F_loss": total_loss_F,
                "D_X_loss": disc_X_loss,
                "D_Y_loss": disc_Y_loss,
            }


    # In[ ]:


    class GANMonitor(keras.callbacks.Callback):
        """A callback to generate and save images after each epoch"""

        def __init__(self, num_img=4,show_result=False,predict_image_save_path='./'):
            self.num_img = num_img
            self.show_result=show_result
            self.predict_image_save_path=predict_image_save_path

        def on_epoch_end(self,epoch,logs=None):
            _, ax = plt.subplots(4, 2, figsize=(12, 12))
            for i, img in enumerate(test_C.take(self.num_img)):
                prediction = self.model.gen_G(img)[0].numpy()
                prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
                img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

                ax[i, 0].imshow(img)
                ax[i, 1].imshow(prediction)
                ax[i, 0].set_title("Input image")
                ax[i, 1].set_title("Translated image")
                ax[i, 0].axis("off")
                ax[i, 1].axis("off")

                prediction = keras.utils.array_to_img(prediction)
                prediction.save(
                    self.predict_image_save_path+"generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
                )

            if self.show_result:
                plt.show()
                plt.close()


    # In[ ]:


    # Create cycle gan model
    cycle_gan_model = CycleGan(
        generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
    )

    # Compile the model
    cycle_gan_model.compile(
        gen_G_optimizer=keras.optimizers.legacy.Adam(learning_rate=args.lr_G_g, beta_1=args.beta1),
        gen_F_optimizer=keras.optimizers.legacy.Adam(learning_rate=args.lr_F_g, beta_1=args.beta1),
        disc_X_optimizer=keras.optimizers.legacy.Adam(learning_rate=args.lr_G_d, beta_1=args.beta1),
        disc_Y_optimizer=keras.optimizers.legacy.Adam(learning_rate=args.lr_F_d, beta_1=args.beta1),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn)


    # In[ ]:


    # Callbacks
    plotter = GANMonitor(show_result=args.show_result,predict_image_save_path=args.predict_image_save_path)
    checkpoint_filepath = args.weights_path+'cyclegan_checkpoints.{epoch:03d}'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, save_weights_only=True
    )

    log_dir = os.path.join(args.tensorboard_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    # In[ ]:

    cycle_gan_model.fit(
        tf.data.Dataset.zip((train_C, train_D)),
        epochs=args.epochs,
        callbacks=[plotter, model_checkpoint_callback,tensorboard_callback],
    )


    # In[ ]:


if __name__ == '__main__':
    main()
    

