import tensorflow as tf
import mahotas
import skimage.io as imgio
import numpy
import scipy.misc
import os

if __name__ == '__main__':
    p_imgscale = 1.0
    # Defines the filter/blur size.
    p_ksize = 4
    # Number of blurring operations.  
    p_kiter = 3
    in_directory = 'data/source'
    out_directory = 'data/blurred'

    # Define the filter / kernel tensor of width p_ksize, height p_ksize, 
    # 3 input channels and 3 output channels (colors). 
    kernel = numpy.zeros([p_ksize, p_ksize, 3, 3])
    for c in range(0,3):
        for xy in range(0,p_ksize):
            kernel[xy,xy,c,c] = 1.0/p_ksize

    for filename in os.listdir(in_directory):
        # Read an image from the file. An image is represented by H x W x 3 tensor,
        # where each point is encoded by R,G,B colors with values from 0 to 255.
        img_raw = mahotas.imread(in_directory + '/' + filename)

        # Downscale the image and normalize color values to range from 0 to 1.0.
        img_base = scipy.misc.imresize(img_raw, p_imgscale)/255.0

        # Start with a blank all black image.
        v_img = tf.Variable(tf.zeros(img_base.shape), name="Unblurred_Image")    

        op_img_resize = tf.reshape(v_img, [-1, img_base.shape[0], 
            img_base.shape[1], img_base.shape[2]])
        pl_kernel = tf.placeholder("float", shape=kernel.shape, name="Kernel")
        op_init = tf.initialize_all_variables()

        # Nested convolution function
        op_convolve = op_img_resize
        for blurStage in range(0,p_kiter):
            op_convolve = tf.nn.conv2d(op_convolve, pl_kernel, 
                strides=[1, 1, 1, 1], padding='SAME')

        # Generate a blurred image by applying convolution to img_base.
        with tf.Session() as session:
            session.run(op_init)
            img_blurred = session.run(op_convolve, feed_dict={
                v_img: img_base, pl_kernel: kernel})
            
        imgio.imsave(out_directory + '/' + filename, img_blurred[0])
