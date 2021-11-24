from PIL import Image

def downscale(image, size):
    image.thumbnail(size, Image.ANTIALIAS)
    return image

def crop(image, size):
    width, height = image.size   # Get dimensions

    left = (width - size)/2
    top = (height - size)/2
    right = (width + size)/2
    bottom = (height + size)/2

    return image.crop((left, top, right, bottom))