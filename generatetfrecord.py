import tensorflow as tf
from object_detection.utils import dataset_util

# Define your dataset paths
train_data_dir = './Raw'
eval_data_dir = 'path_to_evaluation_data_directory'
output_dir = './TFRecords'

# Define your label map
label_map = {
    'zero_holes': 1,
    'one_hole': 2,
    'two_holes': 3
}

def create_tf_example(image_path, label):
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()

    image_format = b'jpg'  # Change if using a different image format
    width, height = 300, 300  # Adjust as per your image dimensions

    # Create TFExample proto
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature([]),  # Fill in with bounding box coordinates if available
        'image/object/bbox/xmax': dataset_util.float_list_feature([]),
        'image/object/bbox/ymin': dataset_util.float_list_feature([]),
        'image/object/bbox/ymax': dataset_util.float_list_feature([]),
        'image/object/class/text': dataset_util.bytes_list_feature([label.encode('utf-8')]),
        'image/object/class/label': dataset_util.int64_list_feature([label_map[label]])
    }))

    return tf_example

def create_tfrecord(data_dir, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    for image_name in os.listdir(data_dir):
        image_path = os.path.join(data_dir, image_name)
        label = image_name.split('.')[0]  # Extract label from filename
        tf_example = create_tf_example(image_path, label)
        writer.write(tf_example.SerializeToString())
    writer.close()

# Create TFRecord files for training and evaluation data
create_tfrecord(train_data_dir, os.path.join(output_dir, 'train.tfrecord'))
create_tfrecord(eval_data_dir, os.path.join(output_dir, 'eval.tfrecord'))
