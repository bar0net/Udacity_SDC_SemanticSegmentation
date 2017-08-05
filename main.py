import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time

alpha = 0.000001

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    # vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    input_image = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name) 
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)   
    layer3_out = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)  
    layer4_out = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)  
    layer7_out = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name) 
    
    return input_image, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    lay3_filters = vgg_layer3_out.shape[-1] # 256
    lay4_filters = vgg_layer4_out.shape[-1] # 512
    lay7_filters = vgg_layer7_out.shape[-1] # 4096
    
    # Matrix shape: [?, ?, ?,4096]
    # Fully Connected using 1x1 Convolution with layer7  output
    lay7_fcn = tf.layers.conv2d(vgg_layer7_out, lay7_filters, (1,1), (1,1), padding='SAME',
                                kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01))
    
    # Cycle 1 - Matrix Shape [?, ?, ?, 512]
    # (1) Upscale previous step using convolution transpose
    lay7_upscale = tf.layers.conv2d_transpose(lay7_fcn, lay4_filters, (4,4), (2,2), padding='SAME',
                                kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01))
    
    # (2) Fully Connected using 1x1 Convolution with Layer4 output 
    lay4_fcn = tf.layers.conv2d(vgg_layer4_out, lay4_filters, (1,1), (1,1), padding='SAME',
                                kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01))
    
    # (3) Add both tensors
    lay4_skip = tf.add(lay7_upscale, lay4_fcn)

    # Cycle 2 - Matrix shape: [?, ?, ?, 256]
    # (1)  Upscale   
    lay4_upscale = tf.layers.conv2d_transpose(lay4_skip, lay3_filters, (4,4), (2,2), padding='SAME',
                                kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01))
    
    # (2) Fully Connected using 1x1 Convolution
    lay3_fcn = tf.layers.conv2d(vgg_layer3_out, lay3_filters, (1,1), (1,1), padding='SAME',
                                kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01))
    
    # (3) Addition
    lay3_skip = tf.add(lay4_upscale, lay3_fcn)
    
    # Output by upscaling last layer - Matrix shape: [?, ?, ?, num_classes]
    net = tf.layers.conv2d_transpose(lay3_skip, num_classes, (16,16), (8,8), padding='SAME',
                                kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01))
    
    return net
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    
    #print(logits.shape)
    #print(labels.shape)
    
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
    loss = tf.reduce_mean(entropy)
    
    # loss, iou_op = tf.metrics.mean_iou(labels=labels, predictions=logits, num_classes=num_classes)
    
    # Using IoU implementation from:
    # http://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html
    #inter = tf.reduce_sum(tf.multiply(logits,labels))
    #union=tf.reduce_sum(tf.subtract(tf.add(logits,labels),tf.multiply(logits,labels)))
    #loss=tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.div(inter,union))
    
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    
    return logits, train_op, loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function    
    t1 = time.clock()
    
    print ("Start training")
    for epoch in range(epochs):   
        num_batch = 0;
        for image, gt_image in get_batches_fn(batch_size):
            
            feed_dict = {
                    input_image: image,
                    correct_label: gt_image,
                    keep_prob: 1.0,
                    learning_rate: alpha
                    }
            
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict = feed_dict)
            
            
            num_batch += 1
            print("Epoch: {}/{}, Batch: {}, Loss {}, Elapsed Time: {}".format(epoch, epochs, num_batch, loss, time.clock() - t1))
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    
    # TODO: Parameter Models
    epochs = 40
    batch_size = 12
    
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results --> Added Data Augmentation inside the helper function
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        net = layers(layer3_out, layer4_out, layer7_out, num_classes)
        
        label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))
        learning_rate = tf.placeholder(dtype=tf.float32)
        
        
        logits, train_op, loss = optimize(net, label, learning_rate, num_classes)
        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
                
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss,
                 input_image, label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits,
                                      keep_prob, input_image)
        
        # OPTIONAL: Apply the trained model to a video
        

if __name__ == '__main__':
    run()
