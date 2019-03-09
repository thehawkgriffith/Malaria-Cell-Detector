import numpy as np
from model import Model
import matplotlib.pyplot as plt
from scipy import misc
import pandas as pd
import csv
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def generate_file():
    with open('file.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['arr', 'label'])
        for _, _, files in os.walk('./cell_images/Uninfected/'):
            for file in files:
                if file[-3:] == "png":
                    csv_writer.writerow(['./cell_images/Uninfected/'+file, '0'])
        for _, _, files in os.walk('./cell_images/Parasitized/'):
            for file in files:
                if file[-3:] == "png":
                    csv_writer.writerow(['./cell_images/Parasitized/'+file, '1'])

def main():
    # generate_file()
    n_classes = 2
    H = 50
    W = 50
    batch_sz = 32
    epochs = 3
    model = Model([20, 50], [128], n_classes)
    model.compile(H, W)
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    data_generator = datagen.flow_from_directory(
        './cell_images/',
        target_size=(50, 50),
        batch_size=32,
        class_mode='binary')
    for e in range(epochs):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in data_generator:
            losses = model.fit(x_batch, y_batch)
            batches += 1
            if batches >= 27558 / 32:
                break
    plt.plot(losses)
    plt.show()

if __name__ == "__main__":
    main()