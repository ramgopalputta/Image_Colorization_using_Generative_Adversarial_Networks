import os
import cv2
import time
import pathlib
import datetime
import numpy as np
import tensorflow as tf
from IPython import display
from tensorflow.keras import layers

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BUFFER_SIZE = 320
BATCH_SIZE = 32
EPOCHS = 50
IMG_SIZE = 128
LEARNING_RATE = 1e-4

# CREATE AUGMENTED IMAGES(FLIP & ROTATE)
def image_augmentation(path):
    # loop through images in files
    for birdImage in os.listdir(path):
        # read image
        name, extension = os.path.splitext(birdImage)
        original = cv2.imread(path + "/" + birdImage)
        # create augmented images
        if "flipped" not in name:
            if "rightRotate" not in name:
                if "leftRotate" not in name:
                    flipped = cv2.flip(original, 1)
                    cv2.imwrite(path + "/" + name + "_flipped" + extension, flipped)
                    rotate = cv2.rotate(original, cv2.cv2.ROTATE_90_CLOCKWISE)
                    cv2.imwrite(path + "/" + name + "_rightRotate" + extension, rotate)
                    rotate = cv2.rotate(original, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                    cv2.imwrite(path + "/" + name + "_leftRotate" + extension, rotate)


#image_augmentation("dataset/training")

# function for tensorflow to load images
# outputs grayscaled image and real image
def load_image(image_file):
    image = tf.io.read_file(image_file)
    color_image = tf.io.decode_jpeg(image)
    color_image = tf.image.resize(color_image, [IMG_SIZE, IMG_SIZE])
    grayscale_image = tf.image.rgb_to_grayscale(color_image)
    color_image = tf.cast(color_image, tf.float32)
    grayscale_image = tf.cast(grayscale_image, tf.float32)
    color_image = (color_image / 127.5) - 1
    grayscale_image = (grayscale_image / 127.5) - 1
    return grayscale_image, color_image


# Define train and test sets
train_dataset = tf.data.Dataset.list_files('dataset/training/*.jpg')
train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.list_files('dataset/testing/*.jpg').map(load_image).batch(BATCH_SIZE)

# Root loss
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# GENERATOR
generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

# compute generator loss
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = binary_cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (100 * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

# create generator model
def make_generator():
    # attempt one, no skip connections
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 1]))
    down_layers = [64, 128, 256, 512, 512, 512, 512]

    # DOWNSAMPLE
    for i,layer in enumerate(down_layers):
        model.add(tf.keras.layers.Conv2D(layer, 4, strides=2, padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

    # UPSAMPLE
    up_layers = [512, 512, 512, 256, 128, 64]
    for i, layer in enumerate(up_layers):
        model.add(tf.keras.layers.Conv2DTranspose(layer, 4, strides=2, padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh'))
    return model
    """

    # Attempt two using skip connections

    # input
    inp = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 1])

    # DOWNSAMPLE
    down = []
    down_layers = [64, 128, 256, 512, 512, 512, 512]
    for i, layer in enumerate(down_layers):
        d = tf.keras.Sequential()
        d.add(tf.keras.layers.Conv2D(layer, 4, strides=2, padding='same', use_bias=False))
        d.add(tf.keras.layers.LeakyReLU())
        down.append(d)

    # UPSAMPLE
    up = []
    up_layers = [512, 512, 512, 256, 128, 64]
    for i, layer in enumerate(up_layers):
        u = tf.keras.Sequential()
        u.add(tf.keras.layers.Conv2DTranspose(layer, 4, strides=2, padding='same', use_bias=False))
        u.add(tf.keras.layers.ReLU())
        up.append(u)

    output = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')

    # build model and add skip connections
    x = inp
    skips = []
    for d in down:
        x = d(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = output(x)

    return tf.keras.Model(inputs=inp, outputs=x)


# DISCRIMINATOR
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

# compute discriminator loss
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = binary_cross_entropy(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = binary_cross_entropy(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

# create discriminator model
def make_discriminator():

    # takes two inputs(generated image and real image || grayscale image and real image)
    input = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 1], name='input_image')
    target = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3], name='target_image')

    x = tf.keras.layers.concatenate([input, target])
    dc1 = tf.keras.layers.Conv2D(32, 4, strides=2, padding='same', use_bias=False)(x)
    dl1 = tf.keras.layers.LeakyReLU()(dc1)
    dc2 = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', use_bias=False)(dl1)
    dl2 = tf.keras.layers.LeakyReLU()(dc2)
    dc3 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', use_bias=False)(dl2)
    dl3 = tf.keras.layers.LeakyReLU()(dc3)
    dc4 = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same', use_bias=False)(dl3)
    dl4 = tf.keras.layers.LeakyReLU()(dc4)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(dl4)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, use_bias=False)(zero_pad1)
    leaky_relu = tf.keras.layers.LeakyReLU()(conv)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1)(zero_pad2)
    return tf.keras.Model(inputs=[input, target], outputs=last)


#################################################################
#################################################################
#################################################################

# initialize generator and discriminator
generator = make_generator()
discriminator = make_discriminator()

# output progress
def generate_images(model, input_img, target_img, step, train=True):
    prediction = model(test_input, training=True)

    if train:
        tf.keras.utils.save_img('progress/' + str(step) + '.png', prediction[0] * 0.5 + 0.5);
    else:
        tf.keras.utils.save_img('report/' + str(step) + '_input.png', input_img[0] * 0.5 + 0.5);
        tf.keras.utils.save_img('report/' + str(step) + '_real.png', target_img[0] * 0.5 + 0.5);
        tf.keras.utils.save_img('report/' + str(step) + '_predicted.png', prediction[0] * 0.5 + 0.5);


# tensorflow train function
@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step)
        tf.summary.scalar('disc_loss', disc_loss, step=step)


# checkpoint to save model progress
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

# RESTORE WEIGHTS
try:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
except:
    pass

# tensorboard log for charts
log_dir = "logs/"
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# train CGAN model
epoch = 0
test_input, test_target = next(iter(test_dataset.take(1)))
start = time.time()
for batch, (input_image, target) in train_dataset.repeat().take(len(train_dataset) * EPOCHS + 1).enumerate():
    if batch % len(train_dataset) == 0 and batch != 0:
        epoch += 1
        print(f"Epoch {epoch} of {EPOCHS} complete. took {time.time() - start:.2f} seconds.")
        start = time.time()
        if epoch % 1 == 0 and epoch != 0:
            generate_images(generator, test_input, test_target, epoch)
        if epoch % 50 == 0 and epoch != 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
    train_step(input_image, target, batch)

    ### Evaluate model on test data after training
    # for i in range(90):
    #    test_input, test_target = next(iter(test_dataset.take(1)))
    #    generate_images(generator, test_input, test_target, i, False)

