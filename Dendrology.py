import os
import cv2
import json
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from google_images_download import google_images_download




class Model_Utils:
    def import_data(path):
        response = google_images_download.googleimagesdownload()

        arguments = json.load(open('download_config.json', "r"))
        for argument in arguments["records"]:
            
            paths = response.download(argument)

            img_dir = path + argument["image_directory"]
            for filename in os.listdir(img_dir):
                try:
                    with Image.open(img_dir + "/" + filename) as im:
                        print('ok')
                except:
                    print(img_dir + "/" + filename)
                    os.remove(img_dir + "/" + filename)
        
        Model_Utils.check(path)

    def check(path, remove=True):
        """
        # https://stackoverflow.com/questions/65438156/tensorflow-keras-error-unknown-image-file-format-one-of-jpeg-png-gif-bmp-re
        def check_images(s_dir, ext_list):
            bad_images=[]
            bad_ext=[]
            s_list= os.listdir(s_dir)
            for klass in s_list:
                klass_path=os.path.join (s_dir, klass)
                print ('processing class directory ', klass)
                if os.path.isdir(klass_path):
                    file_list=os.listdir(klass_path)
                    for f in file_list:               
                        f_path=os.path.join (klass_path,f)
                        index=f.rfind('.')
                        ext=f[index+1:].lower()
                        if ext not in ext_list:
                            print('file ', f_path, ' has an invalid extension ', ext)
                            bad_ext.append(f_path)
                        if os.path.isfile(f_path):
                            try:
                                # https://stackoverflow.com/questions/33548956/detect-avoid-premature-end-of-jpeg-in-cv2-python
                                with open(f_path, 'rb') as f:
                                    check_chars = f.read()[-2:]
                                if check_chars != b'\xff\xd9':
                                    print('file ', f_path, ' is not a valid image file, Not a finished file...')
                                    bad_images.append(f_path)
                                    if remove:
                                        os.remove(f_path)
                                else:
                                    img=cv2.imread(f_path)
                                    shape=img.shape
                            except:
                                print('file ', f_path, ' is not a valid image file')
                                bad_images.append(f_path)
                                if remove:
                                    os.remove(f_path)
                        else:
                            print('*** fatal error, you a sub directory ', f, ' in class directory ', klass)
                else:
                    print ('*** WARNING*** you have files in ', s_dir, ' it should only contain sub directories')
            return bad_images, bad_ext

        good_exts=['jpg', 'jpeg', 'bmp' ] # list of acceptable extensions
        bad_file_list, bad_ext_list=check_images(path, good_exts)
        
        if len(bad_file_list) !=0:
            print("\n\nThe Corrupt files are as follows:")
            for i in range (len(bad_file_list)):
                print (bad_file_list[i])
        else:
            print(' no improper image files were found')
        """
        s_list= os.listdir(path)
        for klass in s_list:
            klass_path=os.path.join (path, klass)
            print ('processing class directory ', klass)
            if os.path.isdir(klass_path):
                file_list=os.listdir(klass_path)
                for f in file_list:
                    f_path=os.path.join (klass_path,f)
                    try:
                        _ = io.imread(f_path)
                        img = cv2.imread(f_path)
                        shape=img.shape

                    except Exception as e:
                        print('file ', f_path, ' is not a valid image file, Not a finished file...')
                        if remove:
                            os.remove(f_path)
                        print(e)

    def get_basic_model():
        # increase input size...
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
        
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
        
            tf.keras.layers.Conv2D(256, (2, 2), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, (2, 2), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
        
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(66)
        ])

        return model



class Basic_CNN:
    def ret_model():
        model = Model_Utils.get_basic_model()
        
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        return model
    
    def Train(path):
        BATCH_SIZE = 20
        
        train_ds, val_ds = Basic_CNN.Split_Datasets(path)
        #train_ds, val_ds = Basic_CNN.Split_Datasets()

        model = Basic_CNN.ret_model()

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=100,
            batch_size = BATCH_SIZE,
            shuffle = True
        )

        model.summary()

        tf.keras.utils.plot_model(
            model,
            to_file="model_2.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96,
        )

        model.save("model_2.h5")

        Basic_CNN.save_images(history)

    def Test(weights, class_names, data_path, batch_to_test):
        # Split Data
        train_ds, _ = Basic_CNN.Split_Datasets(data_path)

        
        model = Basic_CNN.get_basic_model()
        model = load_model(weights)

        # Testing
        Data = list(train_ds)[batch_to_test]
        Predicted = [list(i).index(max(i)) for i in model.predict(Data[0])]

        df = pd.DataFrame({'Actual': Data[1], 'Predicted': Predicted})
        result = df["Actual"].eq(df["Predicted"]).sum() / len(df) * 100
        print(df)
        print(result)

    def Split_Datasets(path):
        train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
            path,
            shuffle=True,
            validation_split=0.2,
            subset="both",
            seed=123,
            image_size=(28, 28),
            batch_size = None
        )

        # Augment Function
        train_ds = train_ds.unbatch()
        scale = tf.keras.layers.Rescaling(scale=1./255)
        def augment_data(x, y):
            x = scale(x)
            images = []
            images.append(x)
            images.append(tf.image.random_brightness(x, 0.2))
            images.append(tf.image.random_contrast(x, 0.5, 2.0))
            images.append(tf.image.random_saturation(x, 0.75, 1.25))
            images.append(tf.image.random_hue(x, 0.1))
            images.append(tf.image.random_flip_left_right(x))
            y = tf.repeat(y, repeats = 6)
            return images, y


        # Train Data
        train_ds = train_ds.map(augment_data).unbatch()
        train_ds = train_ds.batch(100)

        # Val Data
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

        return train_ds, val_ds

    def save_images(history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('CNN Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig("CNN_Accuracy.png")
        # summarize history for loss
        plt.clf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('CNN Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig("CNN_Loss.png")


class Basic_Siamese:
    def ret_model():
        model = Model_Utils.get_basic_model()

        input_a = tf.keras.layers.Input(shape=input_shape)
        input_b = tf.keras.layers.Input(shape=input_shape)
        
        processed_a = model(input_a)
        processed_b = model(input_b)

        # Distance Layer
        Distance_Layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))
        L1_distance = Distance_Layer([encoded_l, encoded_r])

        prediction = Dense(1, activation="sigmoid")(L1_distance)
        siamese_net = Model(inputs=[input_a, input_b], outputs=prediction)

        siamese_net.compile(loss = "binary_crossentropy", optimizer = Adam(0.001, decay=2.5e-4), metrics = ["accuracy"])
        return siamese_net
    
    def Train(path):
        BATCH_SIZE = 20
        
        train_ds, val_ds = Basic_Siamese.Split_Datasets(path)
        #train_ds, val_ds = Basic_Siamese.Split_Datasets()

        model = Basic_Siamese.ret_model()

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=100,
            batch_size = BATCH_SIZE,
            shuffle = True
        )

        model.summary()

        tf.keras.utils.plot_model(
            model,
            to_file="model_2.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96
        )

        model.save("model_2.h5")

        Basic_CNN.save_images(history)
    

    def Split_Datasets(path):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            path,
            shuffle=True,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(28, 28),
            batch_size = None
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            path,
            shuffle=True,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(28, 28),
            batch_size = None
        )

        # Augment Function
        train_ds = train_ds.unbatch()
        scale = tf.keras.layers.Rescaling(scale=1./255)
        def augment_data(x, y):
            x = scale(x)
            images = []
            images.append(x)
            images.append(tf.image.random_brightness(x, 0.2))
            images.append(tf.image.random_contrast(x, 0.5, 2.0))
            images.append(tf.image.random_saturation(x, 0.75, 1.25))
            images.append(tf.image.random_hue(x, 0.1))
            images.append(tf.image.random_flip_left_right(x))
            y = tf.repeat(y, repeats = 6)
            return images, y


        # Train Data
        train_ds = train_ds.map(augment_data).unbatch()
        train_ds = train_ds.batch(100)

        # Val Data
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

        
        def create_pairs(x, digit_indices):
            pairs = []
            labels = []
            
            n=min([len(digit_indices[d]) for d in range(num_classes)]) -1
            
            for d in range(num_classes):
                for i in range(n):
                    z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
                    pairs += [[x[z1], x[z2]]]
                    inc = random.randrange(1, num_classes)
                    dn = (d + inc) % num_classes
                    z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                    pairs += [[x[z1], x[z2]]]
                    labels += [1,0]
            return np.array(pairs), np.array(labels)

        return train_ds, val_ds        

#Model_Utils.import_data("Virginia_Trees/")
#Model_Utils.clean()
#Model_Utils.check("Virginia_Trees/")
Basic_CNN.Train("Virginia_Trees/")
#Basic_CNN.test("model.h5", ["Live_Oak", "Loblolly"], "Virginia_Trees/", 4)