import tensorflow as tf
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas_datareader.data as web
import scipy.optimize as opt
from scipy.optimize import minimize
from datetime import date
import yfinance as yf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU, BatchNormalization
from collections import namedtuple
from FunctionsGANs import discriminator
from FunctionsGANs import generator
from FunctionsGANs import train_step
from FunctionsGANs import prepare_stock_data

disc_model = discriminator()
disc_model.summary()

gen_model = generator()
gen_model.summary()


@tf.function
def train_step(data, batch_size=100):
    noise = tf.random.normal([batch_size, 500])
    # for:
    #    ...
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = gen_model(noise, training=True)
        y_real = disc_model(data, training=True)
        y_fake = disc_model(generated_data, training=True)

        gen_loss = -tf.math.reduce_mean(y_fake)
        disc_loss = tf.reduce_mean(y_fake) - tf.reduce_mean(y_real)

    gradients_gen = gen_tape.gradient(gen_loss, gen_model.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, disc_model.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_gen, gen_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_disc, disc_model.trainable_variables))

    return gen_loss, disc_loss


gen_model = generator()
# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

gen_loss_history = []
disc_loss_history = []

num_batches = x_train.shape[0] // 200
for epoch in range(100):
    for i in tqdm.tqdm(range(num_batches)):
        batch = x_train[i*200:(1+i)*200]
        gen_loss, disc_loss = train_step(batch)

        gen_loss_history.append(gen_loss.numpy())
        disc_loss_history.append(disc_loss.numpy())
        
        
# Graficar pérdidas
plt.plot(gen_loss_history)
plt.plot(disc_loss_history)

# Generar series
noise = tf.random.normal([100, 500])
generated_series = gen_model(noise, training=False)

plt.figure(figsize=(12, 6))
for j in range(100):
    plt.plot(generated_series[j, :])

plt.title("Rendimientos generados")
plt.xlabel("Tiempos")
plt.ylabel("Valores de rendimiento")
plt.legend()
plt.show()