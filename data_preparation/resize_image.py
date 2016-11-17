'''
Created on Oct 27, 2016

@author: vaibhavg
'''
import PIL
import ntpath
from os import listdir
from os.path import isfile, join
from PIL import Image

input_dir = '/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/oxbuild_images/'
onlyfiles = [join(input_dir, f) for f in listdir(input_dir) if isfile(join(input_dir, f))]
for f in onlyfiles:
    print 'Processing ' + f
    img = Image.open(f)
    img = img.resize((256,256), PIL.Image.ANTIALIAS)
    img.save('/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/processed/' + ntpath.basename(f), "JPEG")
    print 'Processed ' + f
