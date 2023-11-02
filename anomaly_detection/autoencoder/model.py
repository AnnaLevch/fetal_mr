from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, concatenate
from keras.models import Model
from keras import backend as K
from training.train_functions.metrics import l1_dice_loss


def context_autoencoder_depth3(input_shape=(256, 256, 5,), loss='binary_crossentropy'):

    input_img = Input(input_shape)  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')(x)

    if(loss == l1_dice_loss):
        loss = loss(int(input_shape[2]/2))

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss=loss)

    return autoencoder


def context_autoencoder_depth4(input_shape=(256, 256, 5,), loss='binary_crossentropy'):

    input_img = Input(input_shape)  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(5, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss=loss)

    return autoencoder


def context_autoencoder(input_shape=(256, 256, 5,),loss='binary_crossentropy'):

    input_img = Input(input_shape)  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(5, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss=loss)

    return autoencoder



def single_slice_autoencoder(input_shape=(256, 256, 1,),loss='binary_crossentropy'):

    input_img = Input(input_shape)  # adapt this if using `channels_first` image data format

    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2), padding='same')(batch1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    batch2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2), padding='same')(batch2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    batch3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2), padding='same')(batch3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    batch4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D((2, 2), padding='same')(batch4)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    batch5 = BatchNormalization()(conv5)
    encoded_pool5 = MaxPooling2D((2, 2), padding='same')(batch5)


    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded_pool5)
    batch6 = BatchNormalization()(conv6)
    up6 = UpSampling2D((2, 2))(batch6)
    merge6 = concatenate([batch5,up6], axis = 3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge6)
    batch7 = BatchNormalization()(conv7)
    up7 = UpSampling2D((2, 2))(batch7)
    merge7 = concatenate([batch4,up7], axis = 3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge7)
    batch8 = BatchNormalization()(conv8)
    up8 = UpSampling2D((2, 2))(batch8)
    conv9 = Conv2D(32, (3, 3), activation='relu',padding='same')(up8)
    batch9 = BatchNormalization()(conv9)
    up9 = UpSampling2D((2, 2))(batch9)
    conv10 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
    batch10 = BatchNormalization()(conv10)
    up10 = UpSampling2D((2, 2))(batch10)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up10)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss=loss)

    return autoencoder


def single_slice_autoencoder_unet(input_shape=(256, 256, 1,), loss='binary_crossentropy'):

    input_img = Input(input_shape)  # adapt this if using `channels_first` image data format

    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2), padding='same')(batch1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    batch2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2), padding='same')(batch2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    batch3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2), padding='same')(batch3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    batch4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D((2, 2), padding='same')(batch4)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    batch5 = BatchNormalization()(conv5)
    encoded_pool5 = MaxPooling2D((2, 2), padding='same')(batch5)


    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded_pool5)
    batch6 = BatchNormalization()(conv6)
    up6 = UpSampling2D((2, 2))(batch6)
    merge6 = concatenate([batch5,up6], axis = 3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge6)
    batch7 = BatchNormalization()(conv7)
    up7 = UpSampling2D((2, 2))(batch7)
    merge7 = concatenate([batch4,up7], axis = 3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge7)
    batch8 = BatchNormalization()(conv8)
    up8 = UpSampling2D((2, 2))(batch8)
    merge8 = concatenate([batch3,up8], axis = 3)
    conv9 = Conv2D(32, (3, 3), activation='relu',padding='same')(merge8)
    batch9 = BatchNormalization()(conv9)
    up9 = UpSampling2D((2, 2))(batch9)
    merge9 = concatenate([batch2,up9], axis = 3)
    conv10 = Conv2D(16, (3, 3), activation='relu', padding='same')(merge9)
    batch10 = BatchNormalization()(conv10)
    up10 = UpSampling2D((2, 2))(batch10)
    merge10 = concatenate([batch1,up10], axis = 3)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(merge10)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss=loss)

    return autoencoder


def context_autoencoder_simple(input_shape=(256, 256, 5,)):

    input_img = Input(input_shape)  # adapt this if using `channels_first` image data format

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(5, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder


def single_slice_autoencoder_simple(input_shape=(256, 256, 1,), loss='binary_crossentropy'):

    input_img = Input(input_shape)  # adapt this if using `channels_first` image data format

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2), padding='same')(batch1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    batch2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2), padding='same')(batch2)
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool2)
    batch3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2), padding='same')(batch3)
    conv4 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool3)
    batch4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D((2, 2), padding='same')(batch4)
    conv5 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool4)
    batch5 = BatchNormalization()(conv5)
    encoded_pool5 = MaxPooling2D((2, 2), padding='same')(batch5)


    conv6 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_pool5)
    batch6 = BatchNormalization()(conv6)
    up6 = UpSampling2D((2, 2))(batch6)
    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same')(up6)
    batch7 = BatchNormalization()(conv7)
    up7 = UpSampling2D((2, 2))(batch7)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(up7)
    batch8 = BatchNormalization()(conv8)
    up8 = UpSampling2D((2, 2))(batch8)
    conv9 = Conv2D(32, (3, 3), activation='relu',padding='same')(up8)
    batch9 = BatchNormalization()(conv9)
    up9 = UpSampling2D((2, 2))(batch9)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    batch10 = BatchNormalization()(conv10)
    up10 = UpSampling2D((2, 2))(batch10)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up10)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss=loss)

    return autoencoder


def single_slice_autoencoder_3pool(input_shape=(256, 256, 1,), loss='binary_crossentropy'):

    input_img = Input(input_shape)  # adapt this if using `channels_first` image data format

    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2), padding='same')(batch1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    batch2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2), padding='same')(batch2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    batch3 = BatchNormalization()(conv3)

    encoded_pool3 = MaxPooling2D((2, 2), padding='same')(batch3)


    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded_pool3)
    batch4 = BatchNormalization()(conv4)
    up4 = UpSampling2D((2, 2))(batch4)
    conv5 = Conv2D(32, (3, 3), activation='relu',padding='same')(up4)
    batch5 = BatchNormalization()(conv5)
    up5 = UpSampling2D((2, 2))(batch5)
    conv6 = Conv2D(16, (3, 3), activation='relu', padding='same')(up5)
    batch6 = BatchNormalization()(conv6)
    up6 = UpSampling2D((2, 2))(batch6)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up6)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss=loss)

    return autoencoder


def single_slice_autoencoder_simple_skip(input_shape=(256, 256, 1,), loss='binary_crossentropy'):

    input_img = Input(input_shape)  # adapt this if using `channels_first` image data format

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2), padding='same')(batch1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    batch2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2), padding='same')(batch2)
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool2)
    batch3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2), padding='same')(batch3)
    conv4 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool3)
    batch4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D((2, 2), padding='same')(batch4)
    conv5 = Conv2D(4, (3, 3), activation='relu', padding='same')(pool4)
    batch5 = BatchNormalization()(conv5)
    encoded_pool5 = MaxPooling2D((2, 2), padding='same')(batch5)


    conv6 = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded_pool5)
    batch6 = BatchNormalization()(conv6)
    up6 = UpSampling2D((2, 2))(batch6)
    merge6 = concatenate([batch5,up6], axis = 3)
    conv7 = Conv2D(8, (3, 3), activation='relu', padding='same')(merge6)
    batch7 = BatchNormalization()(conv7)
    up7 = UpSampling2D((2, 2))(batch7)
    merge7 = concatenate([batch4,up7], axis = 3)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(merge7)
    batch8 = BatchNormalization()(conv8)
    up8 = UpSampling2D((2, 2))(batch8)
    conv9 = Conv2D(32, (3, 3), activation='relu',padding='same')(up8)
    batch9 = BatchNormalization()(conv9)
    up9 = UpSampling2D((2, 2))(batch9)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    batch10 = BatchNormalization()(conv10)
    up10 = UpSampling2D((2, 2))(batch10)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up10)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss=loss)

    return autoencoder