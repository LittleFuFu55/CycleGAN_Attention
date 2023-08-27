# Tensorflow-Keras - CycleGAN_Attention

Tensorflow-Keras implementation of [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593)
by Jun-Yan Zhu et al., and add attention mechanism in generator network.You can find the original CycleGAN implementation from Keras official website
in A_K_Nain [here](https://keras.io/examples/generative/cyclegan/)

## Application - Banana Ripeness Transfer
As in the style transfer case, I use a CycleGAN model to transform the Banana Ripeness, and use an attention mechanism to keep the background style from being transferred as much as possible.

![](Data/ReadME/Predict_result.png)

## Getting Started
### Installation
Code tested in python 3.7.0, Tensorflow 2.10.1 and  Keras 2.10.0 .
Install the requirements with:\
`pip install -r requirements.txt`
### Train CycleGAN_Attention
1. Download your own two domain dataset from google image.
2. Set input images as class C, and set target domain images as class D.
3. Train CycleGAN_Attention
   ```
   python train.py --G_folder_path "class C path" --F_folder_path "class D path" --epochs "training epochs" --weights_path "path to checkpoint"
   ```
4. Implement style transfer task with trained model
   ```
   python predict.py --test_G_folder_path "test class C path" --predict_image_path "fake class D path" --predict_num "Number of predictive images" --weights_path "path to checkpoint"
   ```

### Checking your training with TensorBoard:
 Run TensorBoard:
 ```
 tensorboard --logdir="path to tensorboard folder"
 ```

![](Data/ReadME/G_G_Loss.png) 
![](Data/ReadME/G_F_Loss.png)
![](Data/ReadME/D_X_Loss.png) 
![](Data/ReadME/D_Y_Loss.png)
