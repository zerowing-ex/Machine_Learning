import os
import gzip
import numpy

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
EPOCHS = 20


# Extract the images
def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        # data = numpy.reshape(data, [num_images, -1])
    return data


def extract_labels(filename, num_images, use_one_hot: bool = False):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
        one_hot_encoding = labels
        if use_one_hot:
            num_labels_data = len(labels)
            one_hot_encoding = numpy.zeros((num_labels_data, NUM_LABELS))
            one_hot_encoding[numpy.arange(num_labels_data), labels] = 1
            one_hot_encoding = numpy.reshape(one_hot_encoding, [-1, NUM_LABELS])
        else:
            one_hot_encoding = numpy.expand_dims(one_hot_encoding, -1)
    return one_hot_encoding


def load_dataset():
    pwd: str = os.path.dirname(os.path.realpath(__file__))
    x_train = extract_data(pwd + r'\train-images-idx3-ubyte.gz', 60000)
    y_train = extract_labels(pwd + r'\train-labels-idx1-ubyte.gz', 60000)
    x_valid = extract_data(pwd + r'\t10k-images-idx3-ubyte.gz', 10000)
    y_valid = extract_labels(pwd + r'\t10k-labels-idx1-ubyte.gz', 10000)
    return x_train, y_train, x_valid, y_valid
