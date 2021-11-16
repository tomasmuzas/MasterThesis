import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE

def read_tfrecord(example, image_size):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
        
        # additional (not very useful) fields to demonstrate TFRecord writing/reading of different types of data
        "label":         tf.io.FixedLenFeature([], tf.string),  # one bytestring
        "size":          tf.io.FixedLenFeature([2], tf.int64),  # two integers
        "one_hot_class": tf.io.VarLenFeature(tf.float32)        # a certain number of floats
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)
    
    # FixedLenFeature fields are now ready to use: exmple['size']
    # VarLenFeature fields require additional sparse_to_dense decoding
    
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.reshape(image, [image_size, image_size, 3])
    
    class_num = example['class']
    
    label  = example['label']
    height = example['size'][0]
    width  = example['size'][1]
    one_hot_class = tf.sparse.to_dense(example['one_hot_class'])
    return image, class_num, label, height, width, one_hot_class

@tf.function
def random_invert_horizontally(x, p=0.5):
  if  tf.random.uniform([]) < p:
    x = tf.image.flip_left_right(x)
  else:
    x
  return x

@tf.function
def random_invert_vertically(x, p=0.5):
  if  tf.random.uniform([]) < p:
    x = tf.image.flip_up_down(x)
  else:
    x
  return x

@tf.function
def random_rotate(x, p=0.5):
  if  tf.random.uniform([]) < p:
    x = tf.image.rot90(x, k = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32))
  else:
    x
  return x

RandomHorizontalFlip = tf.keras.layers.Lambda(lambda x: random_invert_horizontally(x))
RandomVerticalFlip = tf.keras.layers.Lambda(lambda x: random_invert_vertically(x))
RandomRotate = tf.keras.layers.Lambda(lambda x: random_rotate(x))

def read_tf_record_dataset(path, name, preprocessing_function, image_size, batch_size, augment = False):
  filenames = tf.io.gfile.glob(path + "/{}/*.tfrec".format(name))
  dataset4 = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
  dataset4 = dataset4.map(lambda x: read_tfrecord(x, image_size), num_parallel_calls=AUTO)
  dataset4 = dataset4.map(lambda image, class_num, label, height, width, one_hot_class: (image, one_hot_class))
  dataset4 = dataset4.map(lambda x, y: (tf.cast(x, tf.float32), y))
  dataset4 = dataset4.map(lambda x, y: (preprocessing_function(x), y))
  if(augment):
    dataset4 = dataset4.map(RandomHorizontalFlip, num_parallel_calls=AUTO)
    dataset4 = dataset4.map(RandomVerticalFlip, num_parallel_calls=AUTO)
    dataset4 = dataset4.map(RandomRotate, num_parallel_calls=AUTO)

  return dataset4.batch(batch_size, drop_remainder=True).prefetch(AUTO)