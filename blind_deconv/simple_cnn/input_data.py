#!/usr/bin/env python

"""Functions for downloading and reading MNIST data."""
import numpy
import scipy.misc as misc
from os import listdir
from os.path import isfile, join
from PIL import Image

labels_dir = '/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/training2/data/source/'
gb_dir = '/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/training2/data/blurred/'

image_width = 64
label_width = 64

def extract_images(folder, num_training, num_test):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    num_tr = 0
    num_tt = 0
    images_tr = []
    images_tt = []
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    for f in onlyfiles:
        if num_tr < num_training:
            images_tr.append(f)
            num_tr += 1
            continue
        if num_tt < num_test:
            images_tt.append(f)
            num_tt += 1
            continue
        break
    return (images_tr, images_tt)

class DataSet(object):
    def __init__(self, images, labels):
        self._num_examples = len(images)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    def _processSingleImage(self, path, resize = None, mode = 'RGB', label = False):
        img = misc.imread(path)
        if resize is not None:
            size = resize[0] / 2
            left = (img.shape[0] / 2) - size
            right = (img.shape[0] / 2) + size
            img = img[left:right, left:right]
            #img = misc.imresize(img, resize, mode=mode)
#         if label:
#             misc.imsave('/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/training/output/bar'+".jpeg", img)
#             im2 = Image.fromarray(img, 'RGB')
#             im2.save('/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/training/output/bar'+".jpeg")
        # Convert from [0, 255] -> [0.0, 1.0].
        img = img.astype(numpy.float32)
#         if label:
#             misc.imsave('/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/training/output/bar1'+".jpeg", img)
        img = numpy.multiply(img, 1.0 / 255.0)
#         if label:
#             misc.imsave('/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/training/output/bar2'+".jpeg", img)
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns*depth]
        if label:
            img = img.reshape(1, img.shape[0] * img.shape[1] * img.shape[2])
            #img = img.reshape(1, img.shape[0] * img.shape[1] * img.shape[2])
#             img2 = img.reshape(256, 256, 3)
#             misc.imsave('/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/training/output/bar3'+".jpeg", img2)
        else:
            img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        return img

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def PrintBatch(self, batch, name):
        labp = batch.reshape(10, 256, 256, 3)
        labp = numpy.split(labp, 10, 0)
        for i in xrange(10):
            lab = labp[i]
            lab = lab.reshape(256, 256, 3)
            misc.imsave('/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/training/output/foo_' + name + str(i) + ".jpeg", lab)
#             lab = Image.fromarray(lab, 'RGB')
#             lab.save('/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/training/output/foo_' + name + str(i) + ".jpeg")
            
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
#             perm = numpy.arange(self._num_examples)
#             numpy.random.shuffle(perm)
#             self._images = self._images[perm]
#             self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        images = None
        labels = None
        for name in self._images[start:end]:
            path = join(gb_dir, name)
            #print 'Processing ' + path
            img = self._processSingleImage(path, (image_width, image_width))
            if images is None:
                images = img
            else:
                images = numpy.concatenate((images, img))
        for name in self._labels[start:end]:
            path = join(labels_dir, name)
            #print 'Processing ' + path
            lbl = self._processSingleImage(path, (label_width, label_width), mode='RGB', label = True)
            if labels is None:
                labels = lbl
            else:
                labels = numpy.concatenate((labels, lbl))
                #print 'labels shape ' + str(labels.shape)
#         self.PrintBatch(images, 'aaa')
#         self.PrintBatch(labels, 'bbb')
        return images, labels


def read_data_sets():
    class DataSets(object):
        pass
    data_sets = DataSets()
    (train_images, test_images) = extract_images(gb_dir, 5000, 10)
    (train_labels, test_labels) = extract_images(labels_dir, 5000, 10)

    data_sets.train = DataSet(train_images, train_labels)
    data_sets.test = DataSet(test_images, test_labels)
    return data_sets
