import random
from PIL import Image
from PIL import ImageFilter

def GaussianBlurImage(input_path, output_path):
    im = Image.open(input_path)
    im = im.filter(ImageFilter.GaussianBlur(4))
    im.save(output_path, 'JPEG')

def BlackenPixels(input_path, output_path, fraction):
    random.seed(3)
    im = Image.open(input_path)
    pixels = im.load()
    (width, height) = im.size
    for i in range(width):    # for every pixel:
        for j in range(height):
            value = random.random()
            if value < fraction:
                pixels[i,j] = (0, 0, 0) # Black
    im.save(output_path, 'JPEG')

if __name__ == '__main__':
    input_path = '/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/processed/oxford_002805.jpg'
    test_path = '/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/test/oxford_002805_bp.jpg'
    BlackenPixels(input_path, test_path, 0.2)
    
