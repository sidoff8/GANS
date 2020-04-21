
import mlflow
import tensorflow as tf
print(tf.__version__)
import helper
helper.download_extract('celeba', 'data/')


# In[2]:
import os
from glob import glob
import numpy as np
from matplotlib import pyplot
show_n_images = 25
data_dir='data/'

celeba_images = helper.get_batch(glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))[:show_n_images], 28, 28, 'RGB')
pyplot.imshow(helper.images_square_grid(celeba_images, 'RGB'))


# In[3]:


celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))


# In[4]:


celeba_dataset.shape


# In[5]:


exec('from __future__ import absolute_import, division, print_function')


# In[6]:


import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import PIL
import tensorflow.keras.layers as layers
import time
import sys

from IPython import display


# In[7]:
with mlflow.start_run():
    
    
   # for name, value in PARAMS.items():
    learning_rate=sys.argv[1]
    epochs=sys.argv[2]
    noise_dim=sys.argv[3]
    num_examples_to_generate=sys.argv[4]
    buffer_size=sys.argv[5]
    batch_size=sys.argv[6]
    mlflow.log_param('learning_rate',learning_rate)
    mlflow.log_param('epochs',epochs)
    mlflow.log_param('noise_dim',noise_dim)
    mlflow.log_param('num_examples_to_generate',num_examples_to_generate)
    mlflow.log_param('buffer_size',buffer_size)
    mlflow.log_param('batch_size',batch_size)
    PARAMS={
    'learning_rate' : float(learning_rate),
    'epochs' : int(epochs),
    'noise_dim' : int(noise_dim),
    'num_examples_to_generate' : int(num_examples_to_generate),
    'buffer_size' : int(buffer_size),
    'batch_size' : int(batch_size)
    }
    def make_generator_model():
        model = tf.keras.Sequential()
        model.add(layers.Dense(7*7*512, use_bias=False, input_shape=(PARAMS['noise_dim'],)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 512)))
        assert model.output_shape == (None, 7, 7, 512) 
        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 256)  
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 128)    
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 3)

        return model


    # In[8]:


    generator = make_generator_model()

    noise = tf.random.normal([1, PARAMS['noise_dim']])
    #noise = tf.placeholder(tf.float32, [None, 100], 'input_z')
    generated_image = generator(noise,training=False)
    plt.imshow(generated_image[0, :, :, :],cmap='gray')


    # In[9]:


    def make_discriminator_model():
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, 5, strides=(2, 2), padding='same', 
                                         input_shape=[28, 28, 3]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(128, 5, strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(256, 5, strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model


    # In[10]:


    discriminator = make_discriminator_model()
    decision = discriminator(generated_image)
    print (decision)


    # In[11]:


    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    # In[12]:


    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


    # In[13]:


    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)


    # In[14]:


    generator_optimizer = tf.keras.optimizers.Adam(PARAMS['learning_rate'])
    discriminator_optimizer = tf.keras.optimizers.Adam(PARAMS['learning_rate'])


    # In[15]:


    checkpoint_dir = 'training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)


    # In[16]:


    #EPOCHS = 5
    #noise_dim = 100
    #num_examples_to_generate = 16

    seed = tf.random.normal([PARAMS['num_examples_to_generate'], PARAMS['noise_dim']])


    # In[17]:


    @tf.function
    def train_step(images):
        noise = tf.random.normal([PARAMS['batch_size'], PARAMS['noise_dim']])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          generated_images = generator(noise, training=True)

          real_output = discriminator(images, training=True)
          fake_output = discriminator(generated_images, training=True)

          gen_loss = generator_loss(fake_output)
          disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        
        mlflow.log_param("Generator Loss", gen_loss)
        mlflow.log_param("Discrminator Loss", disc_loss)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


    # In[18]:


    def train( get_batches, data_shape, data_image_mode, epochs, batch_size=64):  
      for epoch in range(epochs):
        start = time.time()

        for image_batch in get_batches(batch_size):
          train_step(image_batch)


        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)


        if (epoch + 1) % 3 == 0:
          checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


      display.clear_output(wait=True)
      generate_and_save_images(generator,
                               epochs,
                               seed)


    # In[19]:


    def generate_and_save_images(model, epoch, test_input):
        predictions = model(test_input, training=False)
        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')
        plt.savefig('img/image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()


    # In[21]:


    #BUFFER_SIZE = 202599
    #BATCH_SIZE = 256


    # In[ ]:

    
    train(celeba_dataset.get_batches,
              celeba_dataset.shape, celeba_dataset.image_mode, PARAMS['epochs'],PARAMS['batch_size'])
