import os
import sys
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

#tf.debugging.set_log_device_placement(True)

epochs = 10
train_data_dir = './data/train/'
validation_data_dir = './data/validation/'

def tester():
    img_width, img_height = 356, 295#489, 405  #최대 약 350x350 크기... vram 부족...
    train_samples = 7836
    validation_samples = 2000
    filters1 = 64 #32
    filters2 = 64 #32
    filters3 = 128 #64
    conv1_size = 5 #3
    conv2_size = 3 #2
    conv3_size = 7 #5
    pool_size = 2
    # We have 2 classes, long and short
    classes_num = 2
    batch_size = 32  # 128

    #1
    model = models.Sequential()
    model.add(layers.Conv2D(filters1, (conv1_size, conv1_size), padding='same',  input_shape=(img_height, img_width , 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(layers.Dropout(0.25))
    #2
    model.add(layers.Conv2D(filters2, (conv2_size, conv2_size), padding="same"))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(pool_size, pool_size),data_format='channels_last'))
    model.add(layers.Dropout(0.25))
    #5
    model.add(layers.Conv2D(filters3, (conv3_size, conv3_size), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(pool_size, pool_size),data_format='channels_last'))
    model.add(layers.Dropout(0.25))
    #1024 neuron hidden layer
    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes_num, activation='softmax'))#10

    model.summary()

    #모델 컴파일
    #sgd = tf.keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=["acc"])#accuracy
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy', metrics=['acc'])#lr=0.0001, decay=1e-6
    #model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, decay=1e-6, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])#SGD
    #model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['acc'])

    train_datagen = ImageDataGenerator(#rescale=1. / 255,
                                       horizontal_flip=False)
    test_datagen = ImageDataGenerator(#rescale=1. / 255,
                                      horizontal_flip=False)
    #validation_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=False)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_height, img_width),
             shuffle=True,
            batch_size=batch_size,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
             shuffle=True,
            class_mode='categorical')


    #체크포인트
    metric = 'val_acc'
    target_dir = "./models/weights-improvement/"
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    model.save('./models/model.h5')
    model.save_weights('./models/weights.h5')

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath= target_dir + 'weights-improvement-{epoch:02d}-{acc:.2f}.h5',#
                                                    monitor=metric, verbose=2, save_best_only=True, mode='max')
    #텐서보드 사용
    #logdir = "./logs/" +datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


    #모델 훈련
    #model.fit(train_images, train_labels, epochs=10)

    callbacks_list = [checkpoint]

    #모델 불러오기
    #if(target_dir):
    #    model.load_weights(target_dir)

    model.fit(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        shuffle=True,
        validation_data=validation_generator,
        callbacks=callbacks_list,#[checkpoint],
        #callbacks=[tensorboard_callback],#텐서보드
        validation_steps=validation_samples // batch_size)

    #모델 평가
    #
    #test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    #print(test_acc)
    model.save('./models/model.h5')
    model.save_weights('./models/weights.h5')


############################################################################

if __name__ == '__main__':
    #1
    #physical_devices = tf.config.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #2
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #if gpus:
    #    try:
    #        tf.config.experimental.set_memory_growth(gpus[0], True)
    #    except RuntimeError as e:
    #        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    #        print(e)

    tester()


