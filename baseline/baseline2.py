# source ~/tensorflow/bin/activate
# python ~/Desktop/Stanford/eclipse_workspace/CS221-Project/baseline.py
import tensorflow as tf
import mahotas
import skimage.io as imgio
import ntpath
from os import listdir
from os.path import isfile, join
import numpy
import scipy.misc
import math
import random
from PIL import Image
from PIL import ImageFilter


if __name__ == '__main__':
    training_dir = '/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/training/input_bp/'
    labels_dir = '/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/training/input_labels/'
    output_dir = '/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/training/output/'
    onlyfiles = [join(training_dir, f) for f in listdir(training_dir) if isfile(join(training_dir, f))]
    count = 0
    mses = []
    psnrs = []
    for p_imgfile in onlyfiles:
        count += 1
        if count > 10:
            break
        img_label = mahotas.imread(labels_dir + ntpath.basename(p_imgfile))
        img_label = img_label/255.0
        img_blurred = mahotas.imread(p_imgfile)
        img_blurred = img_blurred/255.0
        
        
        def f_pixel(x):
            return 0 if x<0 else 1 if x>1 else x
        f_img = numpy.vectorize(f_pixel, otypes=[numpy.float])
        
        p_ksize = 4
        p_kiter = 3
        kernel = numpy.zeros([p_ksize, p_ksize, 3, 3])
        for c in range(0,3):
            for xy in range(0,p_ksize):
                kernel[xy,xy,c,c] = 1.0/(p_ksize)
        pl_kernel = tf.placeholder("float", shape=kernel.shape, name="Kernel")
        v_img = tf.Variable(tf.zeros(img_label.shape), name="Unblurred_Image")
        op_img_resize = tf.reshape(v_img, [-1, img_label.shape[0], 
            img_label.shape[1], img_label.shape[2]])
        op_convolve = op_img_resize
        for blurStage in range(0,p_kiter):
            op_convolve = tf.nn.conv2d(op_convolve, pl_kernel, 
                strides=[1, 1, 1, 1], padding='SAME')
        pl_blurredImg = tf.placeholder("float", shape=img_blurred.shape)
        op_loss = tf.reduce_sum(tf.square(op_convolve - pl_blurredImg))
        op_optimize = tf.train.GradientDescentOptimizer(0.5).minimize(op_loss)
        
        op_init = tf.initialize_all_variables()
        with tf.Session() as session:
            session.run(op_init)
            for epoch in range(0,2):
                img_deblurred = f_img(session.run(v_img, feed_dict={
                    pl_blurredImg: img_blurred, pl_kernel: kernel}))
                #imgio.imsave("deblurred-%s.png" % epoch, img_deblurred)
                for iteration in range(0,2):
                    error = session.run([op_optimize, op_loss], feed_dict={
                        pl_blurredImg: img_blurred, pl_kernel: kernel})[1]
                    #print("%s/%s = %s" % (epoch, iteration, error))
            img_deblurred = f_img(session.run(v_img, feed_dict={
                pl_blurredImg: img_blurred, pl_kernel: kernel}))
        imgio.imsave(join(output_dir, 'delurred_' + ntpath.basename(p_imgfile)), img_deblurred)
        
        # Converting before getting the diff.
        img_label = img_label * 255
        img_deblurred = img_deblurred * 255
        err = numpy.sum((img_label.astype("float") - img_deblurred.astype("float")) ** 2)
        err /= float(img_deblurred.shape[0] * img_deblurred.shape[1])
        mses.append(err)
        def psnr(img1, img2):
            mse = numpy.mean( (img1 - img2) ** 2 )
            if mse == 0:
                return 100
            PIXEL_MAX = 255.0
            return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        psnr = psnr(img_label, img_deblurred)
        psnrs.append(psnr)
        print 'MSE ' + str(err)
        print 'PSNR ' + str(psnr)
    print 'Mean MSE ' + str(numpy.mean(mses))
    print 'Std MSE ' + str(numpy.std(mses))
    print 'Mean PSNR ' + str(numpy.mean(psnrs))
    print 'Std PSNR ' + str(numpy.std(psnrs))
