import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, Model
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

from matplotlib import pyplot as plt

from sklearn.utils import shuffle

# Current directory path
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Change this to the location of the database directories
DB_DIR = os.path.dirname(os.path.realpath(__file__))

# Import databases
sys.path.insert(1, DB_DIR)

def plot_images(images, show=True, save=False, name='savepoint'):

    images = np.squeeze(images)

    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i])

    if show:
        plt.show()

    if save:
        plt.savefig(name)

    plt.close()

class GAN_model(tf.keras.Model):
    """Generative Adversarial Network for digit generation (MNIST)."""

    def __init__(self, input_shape=(28, 28, 1), rand_vector_shape=(100,), lr=0.0002, beta=0.5):
        super(GAN_model, self).__init__()
        # Input sizes
        self.img_shape = input_shape
        self.input_size = rand_vector_shape

        # optimizers
        self.opt = tf.keras.optimizers.Adam(lr*2, beta)
        self.opt_d = tf.keras.optimizers.Adam(lr, beta)


        # Create generator and discriminator models:
        self.generator = self.generator_model()
        self.generator.compile(loss='binary_crossentropy',
                               optimizer=self.opt,
                               metrics=['accuracy'])

        self.discriminator = self.discriminator_model()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.opt_d,
                                   metrics=['accuracy'])

        # Set the Discriminator as non trainable in the combined GAN model
        self.discriminator.trainable = False

        # Define model input and output
        input = tf.keras.Input(self.input_size)
        generated_img = self.generator(input)
        output = self.discriminator(generated_img)

        # Define and compile combined GAN model
        self.GAN = Model(input, output, name="GAN")

        self.GAN.compile(loss='binary_crossentropy', optimizer=self.opt,
                         metrics=['accuracy'])


    def discriminator_model(self):
        """Create discriminator model."""
        model = models.Sequential(name='Discriminator')
        model.add(layers.Flatten(input_shape=self.img_shape))
        model.add(layers.Dense(200))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dense(100))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def generator_model(self):
        """Create generator model."""
        model = models.Sequential(name='Generator')
        model.add(layers.Dense(256, input_shape=self.input_size))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Dense(1024))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.BatchNormalization(momentum=0.8))

        model.add(layers.Dense(np.prod(self.img_shape)))
        model.add(layers.Reshape(self.img_shape))

        return model


    def generator_model_Youtube(self):
        model = models.Sequential(name='Generator_Youtube')

        model.add(layers.Dense(256, input_shape=(100,)))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Dense(1024))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))

        model.add(layers.Dense(np.prod((28,28,1)), activation='tanh'))
        model.add(layers.Reshape((28,28,1)))

        return model


    def discriminator_model_YOUTUBE(self):
        model = models.Sequential(name='Discriminator_Youtube')

        model.add(layers.Flatten(input_shape=(28, 28, 1)))
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def create_generator(self):
        self.generator = self.generator_model()
        self.generator.compile(loss='binary_crossentropy',
                               optimizer=self.opt_g,
                               metrics=['accuracy'])

        print('created generator model')
        # print(self.generator.summary())

    def create_discriminator(self):
        # Create Discriminator model
        self.discriminator = self.discriminator_model()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.opt_d,
                                   metrics=['accuracy'])

        print('created discriminator model')
        # print(self.discriminator.summary())


    def create_GAN(self):
        '''Train both the Discriminator and the Generator'''

        # Set the Discriminator as non trainable in the combined GAN model
        self.discriminator.trainable = False

        # Define model input and output
        input = tf.keras.Input(self.input_size)
        generated_img = self.generator(input)
        output = self.discriminator(generated_img)

        # Define and compile combined GAN model
        self.GAN = Model(input, output, name="GAN")

        self.GAN.compile(loss='binary_crossentropy', optimizer=self.opt,
                         metrics=['accuracy'])

        # print(self.GAN.summary())


    def train(self, X_train, batch_size=128, epochs=4000, save_interval=100):
        '''Train GAN model'''

        half_batch = int(batch_size/2)
        y_pos_train_dis = np.ones(half_batch)
        y_neg_train_dis = np.zeros(half_batch)
        y_train_GAN = np.ones(batch_size)

        c = 0
        for epoch in range(epochs):

            # Generate training data for discriminator
            X_pos_train_dis = X_train[np.random.randint(0, len(X_train[0]), half_batch)]
            random_vector = np.random.randn(half_batch, 100)                         # OJO!
            X_neg_train_dis = self.generator.predict(random_vector)

            X_train_dis = np.concatenate((X_pos_train_dis, X_neg_train_dis))
            y_train_dis = np.concatenate((y_pos_train_dis, y_neg_train_dis))

            X_train_dis, y_train_dis = shuffle(X_train_dis, y_train_dis)

            # Generate training data for combined GAN model
            X_train_GAN = np.random.randn(batch_size, 100)                           # OJO!

            # Train Discriminator
            self.discriminator.trainable = True
            loss_dis = self.discriminator.fit(X_train_dis, y_train_dis, verbose=0)

            # Train Generator
            self.discriminator.trainable = False
            loss_GAN = self.GAN.fit(X_train_GAN, y_train_GAN, verbose=0)

            # Print results
            if epoch % save_interval is 0:
                c+=1
                print('epoch ', c)
                print('discriminator: ', loss_dis.history)
                print('GAN: ', loss_GAN.history)

                plot_images(X_neg_train_dis[np.random.randint(0, len(X_neg_train_dis[0]), 25)], False, save=True,
                            name= DIR_PATH + '/checkpoints/savepoint_%i.png' % c)



def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

    # load dataset, normalize and reshape
    (X_train, _), (_, _) = mnist.load_data()

    X_train = X_train / 255
    X_train = np.expand_dims(X_train, axis=3)

    # Create GAN model
    GAN = GAN_model()

    # print(GAN.GAN.summary())

    if 'checkpoints' not in os.listdir(DIR_PATH):
        os.mkdir(DIR_PATH + '/checkpoints')

    GAN.train(X_train)


if __name__ == '__main__':
    main()
