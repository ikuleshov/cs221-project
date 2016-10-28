import tensorflow as tf
import mahotas
import skimage.io as imgio
import numpy
import scipy.misc

if __name__ == '__main__':
    p_imgfile = 'monkey.jpg'
    p_imgscale = .33
    # Defines the filter/blur size.
    p_ksize = 4
    # Number of blurring operations.  
    p_kiter = 3

    # Read an image from the file. An image is represented by H x W x 3 tensor,
    # where each point is encoded by R,G,B colors with values from 0 to 255.
    img_raw = mahotas.imread(p_imgfile)

    # Downscale the image and normalize color values to range from 0 to 1.0.
    img_base = scipy.misc.imresize(img_raw, p_imgscale)/255.0

    # Save the pre-processed image as a baseline. 
    imgio.imsave('base.png', img_base)

    # Define the filter / kernel tensor of width p_ksize, height p_ksize, 
    # 3 input channels and 3 output channels (colors). 
    kernel = numpy.zeros([p_ksize, p_ksize, 3, 3])
    for c in range(0,3):
        for xy in range(0,p_ksize):
            kernel[xy,xy,c,c] = 1.0/p_ksize

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
        
    imgio.imsave('blurred.png', img_blurred[0])

    pl_blurredImg = tf.placeholder("float", shape=img_blurred.shape)

    # The loss function is the min squared difference between the
    # convolved unblurred image and the blurred image.
    op_loss = tf.reduce_sum(tf.square(op_convolve - pl_blurredImg))

    # Gradient descent linear regression using the loss function above, 
    # minimizing by the variable v_img (unblurred image). 
    op_optimize = tf.train.GradientDescentOptimizer(0.5).minimize(op_loss, var_list=[v_img])
    
    # Only allow values between [0,1] in the resulting unblurred image.
    def f_pixel(x):
        return 0 if x<0 else 1 if x>1 else x
    f_img = numpy.vectorize(f_pixel, otypes=[numpy.float])

    with tf.Session() as session:
        session.run(op_init)
        for epoch in range(0,5):
            # This is just a checkpoint to show intermediary results and can be omitted.
            img_deblurred = f_img(session.run(v_img, feed_dict={
                pl_blurredImg: img_blurred, pl_kernel: kernel}))
            imgio.imsave("deblurred-%s.png" % epoch, img_deblurred)

            # Run the optimizer by providing the blurred image, kernel which will 
            # update v_img (unblurred image variable).
            for iteration in range(0,10):
                error = session.run([op_optimize, op_loss], feed_dict={
                    pl_blurredImg: img_blurred, pl_kernel: kernel})[1]
                print("%s/%s = %s" % (epoch, iteration, error))

        img_deblurred = f_img(session.run(v_img, feed_dict={
            pl_blurredImg: img_blurred, pl_kernel: kernel}))

    imgio.imsave("deblurred-final.png", img_deblurred)
