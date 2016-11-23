import tensorflow as tf
import mahotas
import skimage.io as imgio
import numpy
import scipy.misc
import os
import random
import math

if __name__ == '__main__':
    truth_directory = 'data/source'
    blurred_directory = 'data/blurred'
    batch_size = 1
    input_size = 32
    output_size = 20

    blurred_img_scale = 1.0
    # Truth image needs to be downsized for training since the predictor's output dimension is smaller than input. 
    truth_img_scale = output_size / float(input_size)

    samples_count = 100
    training_iters = 10000
    display_step = 1000
    learning_rate = 0.001

    random.seed(3)

    def psnr(img1, img2):
        img1 = numpy.multiply(img1, 255.0)
        img2 = numpy.multiply(img2, 255.0)
        mse = numpy.mean( (img1 - img2) ** 2 )
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    # Create model
    def conv_net(x):
        conv_layers = [ (9, 1), (1,64), (5, 32), (1, 3)]
        # First layer has 3 input channels for R, G, B 
        in_channels = 3
        for filter_size, num_filters in conv_layers:
            with tf.name_scope("conv-%s" % filter_size):
                filter_shape = [filter_size, filter_size, in_channels, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # Set the number of input channels for the next layer equal to the number of out channels for the current layer.
                in_channels = num_filters
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # Apply nonlinearity
                x = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        return x

    # tf Graph input
    x = tf.placeholder(tf.float32, [input_size, input_size, 3])
    y = tf.placeholder(tf.float32, [output_size, output_size, 3])

    # Construct model
    pred = conv_net(tf.expand_dims(x, 0))

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.l2_loss(tf.sub(pred, y)))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    accuracy = tf.div(tf.sqrt(tf.mul(tf.reduce_mean(tf.nn.l2_loss(tf.sub(pred, tf.expand_dims(y, 0)))), 2.0)), batch_size)

    op_init = tf.initialize_all_variables()

    # Only allow values between [0,1] in the resulting unblurred image.
    def f_pixel(x):
        return 0 if x<0 else 1 if x>1 else x
    f_img = numpy.vectorize(f_pixel, otypes=[numpy.float])

    # Generate a blurred image by applying convolution to img_base.
    with tf.Session() as session:
        session.run(op_init)

        for filename in os.listdir(truth_directory):
            print "Now training on: ", filename
            filebase = filename[:-4]

            # Read an image from the file. An image is represented by H x W x 3 tensor,
            # where each point is encoded by R,G,B colors with values from 0 to 255.
            truth_img_raw = mahotas.imread(truth_directory + "/" + filename)
            # Downscale the image and normalize color values to range from 0 to 1.0.
            truth_img = scipy.misc.imresize(truth_img_raw, truth_img_scale)/255.0
            imgio.imsave('tmp/%s_truth.png' % filebase, f_img(truth_img))

            blurred_img_raw = mahotas.imread(blurred_directory + "/" + filename)
            blurred_img = scipy.misc.imresize(blurred_img_raw, blurred_img_scale)/255.0

            patch_count_x = blurred_img.shape[1] / input_size
            patch_count_y = blurred_img.shape[0] / input_size
            for i in range(samples_count):
                # Randomly sample a non-overlapping patch from the image.
                # TODO: 
                # 1) Make sure patches are unique.
                # 2) Consider Gaussian distribution to get more samples from the center.
                # 3) Allow overlapping?  
                patch_coord_y = random.randint(0, patch_count_y - 1)
                patch_coord_x = random.randint(0, patch_count_x - 1)

                blurred_patch = blurred_img[ input_size * patch_coord_y: input_size * (patch_coord_y + 1),
                                         input_size * patch_coord_x: input_size * (patch_coord_x + 1)]
                imgio.imsave('tmp/%s_patch_%d_blurred.png' % (filebase, i), f_img(blurred_patch))

                truth_patch = truth_img[ output_size * patch_coord_y: output_size * (patch_coord_y + 1),
                                         output_size * patch_coord_x: output_size * (patch_coord_x + 1)]

                imgio.imsave('tmp/%s_patch_%d_truth.png' % (filebase, i), f_img(truth_patch))
                # Keep training until reach max iterations
                for step in range(training_iters):
                    session.run(optimizer, feed_dict={x: blurred_patch, y: truth_patch})

                    if step > 0 and (step % display_step == 0):
                        # Calculate batch loss and accuracy
                        loss, acc, deblurred_patch = session.run([cost, accuracy, pred], feed_dict={x: blurred_patch, y: truth_patch})
                        print "PSNR:", psnr(deblurred_patch[0], truth_patch)

                        print("Iteration " + str(step) + ", Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))
                        
                        imgio.imsave('tmp/%s_patch_%d_iteration_%d.png' % (filebase, i, step), f_img(deblurred_patch[0]))

    