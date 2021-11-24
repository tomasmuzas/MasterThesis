import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE

def read_tfrecord(example, image_size):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
        
        # additional (not very useful) fields to demonstrate TFRecord writing/reading of different types of data
        "label":         tf.io.FixedLenFeature([], tf.string),  # one bytestring
        "objid":         tf.io.FixedLenFeature([], tf.string),  # two integers
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
    objid = example['objid']
    one_hot_class = tf.sparse.to_dense(example['one_hot_class'])
    return image, class_num, label, objid, one_hot_class

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

def read_tf_record_dataset(path, preprocessing_function, image_size, batch_size, augment = False, include_objid = False):
  filenames = tf.io.gfile.glob(path + "/*.tfrec")
  dataset4 = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
  dataset4 = dataset4.map(lambda x: read_tfrecord(x, image_size), num_parallel_calls=AUTO)
  if (include_objid):
    dataset4 = dataset4.map(lambda image, class_num, label, objid, one_hot_class: (image, class_num, objid))
    dataset4 = dataset4.map(lambda x, y, z: (tf.cast(x, tf.float32), y, z), num_parallel_calls=AUTO)
    dataset4 = dataset4.map(lambda x, y, z: (preprocessing_function(x), y, z), num_parallel_calls=AUTO)
    if(augment):
      dataset4 = dataset4.map(lambda x,y,z : (random_invert_horizontally(x), y, z), num_parallel_calls=AUTO)
      dataset4 = dataset4.map(lambda x,y,z : (random_invert_vertically(x), y, z), num_parallel_calls=AUTO)
      dataset4 = dataset4.map(lambda x,y,z : (random_rotate(x), y, z), num_parallel_calls=AUTO)
  else:
    dataset4 = dataset4.map(lambda image, class_num, label, objid, one_hot_class: (image, class_num))
    dataset4 = dataset4.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTO)
    dataset4 = dataset4.map(lambda x, y: (preprocessing_function(x), y), num_parallel_calls=AUTO)
    if(augment):
      dataset4 = dataset4.map(lambda x,y : (random_invert_horizontally(x), y), num_parallel_calls=AUTO)
      dataset4 = dataset4.map(lambda x,y : (random_invert_vertically(x), y), num_parallel_calls=AUTO)
      dataset4 = dataset4.map(lambda x,y : (random_rotate(x), y), num_parallel_calls=AUTO)

  return dataset4.batch(batch_size, drop_remainder=True).prefetch(AUTO)