import os.path
import tensorflow as tf
import warnings
from distutils.version import LooseVersion
#import project_tests as tests
import csv
import time
import utils
import numpy as np


model_path='./model/model.ckpt'

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.' \
                                                            '  You are using {}'.format(tf.__version__)
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
    # Define the name of the tensors
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Get the needed layers' outputs for building FCN-VGG16
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    image_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out



def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    kernel_regularizer = tf.contrib.layers.l2_regularizer(0.5)
    # Compute logits
    layer3_logits = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=[1, 1],
                                     padding='same', kernel_regularizer=kernel_regularizer)
    layer4_logits = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=[1, 1],
                                     padding='same', kernel_regularizer=kernel_regularizer)
    layer7_logits = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=[1, 1],
                                     padding='same', kernel_regularizer=kernel_regularizer)

    # Add skip connection before 4th and 7th layer
    layer7_logits_up = tf.image.resize_images(layer7_logits, size=[10, 36])
    layer_4_7_fused = tf.add(layer7_logits_up, layer4_logits)

    # Add skip connection before (4+7)th and 3rd layer
    layer_4_7_fused_up = tf.image.resize_images(layer_4_7_fused, size=[20, 72])
    layer_3_4_7_fused = tf.add(layer3_logits, layer_4_7_fused_up)

    # resize to original size
    layer_3_4_7_up = tf.image.resize_images(layer_3_4_7_fused, size=[160, 576])
    layer_3_4_7_up = tf.layers.conv2d(layer_3_4_7_up, num_classes, kernel_size=[15, 15],
                                      padding='same', kernel_regularizer=kernel_regularizer)

    return layer_3_4_7_up



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
    # make logits a 2D tensor where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1,num_classes))
    # define loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))
    # define training operation
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss



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
    # Create log file
    log_filename = "./training_progress.csv"
    log_fields = ['learning_rate', 'exec_time (s)', 'training_loss']
    log_file = open(log_filename, 'w')
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()


    sess.run(tf.global_variables_initializer())

    lr = 0.0001

    print("Training...")
    print()
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        training_loss = 0
        training_samples = 0
        starttime = time.clock()
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label,
                                          keep_prob: 0.8, learning_rate: lr})
            print("batch loss: = {:.3f}".format(loss))
            training_samples += 1
            training_loss += loss

        training_loss /= training_samples
        endtime = time.clock()
        training_time = endtime-starttime

        print("Average loss for the current epoch: = {:.3f}\n".format(training_loss))
        log_writer.writerow({'learning_rate': lr, 'exec_time (s)': round(training_time, 2) , 'training_loss': round(training_loss,4)})
        log_file.flush()





def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    #tests.test_for_kitti_dataset(data_dir)

   

    with tf.Session() as sess:
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = utils.create_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        # TODO: Build NN using load_vgg, layers, and optimize function
        epochs = 30
        batch_size = 8
        

        # TF placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
       #imgMean = np.array([104, 117, 124], np.float)
        #x = tf.placeholder("float", [1, 224, 224, 3])
        #model = vgg19.VGG19(x, dropoutPro, num_classes, skip)
        #score = model.fc8
	    #softmax = tf.nn.softmax(score)
        #softmax = tf.nn.softmax(score)
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     model.loadModel(sess)

        #     for key,img in testImg.items():
        #         #img preprocess
        #         resized = cv2.resize(img.astype(np.float), (224, 224)) - imgMean
        #         maxx = np.argmax(sess.run(softmax, feed_dict = {x: resized.reshape((1, 224, 224, 3))}))
        #         res = caffe_classes.class_names[maxx]

        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         cv2.putText(img, res, (int(img.shape[0]/3), int(img.shape[1]/3)), font, 1, (0, 255, 0), 2)
        #         print("{}: {}\n----".format(key,res))
        #         cv2.imshow("demo", img)
        #         cv2.waitKey(0)
	

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        input_image = tf.get_default_graph().get_tensor_by_name('image_input:0')
        #TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,correct_label, keep_prob, learning_rate)

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        print("Model is saved to file: %s" % save_path)

        # TODO: predict the testing data and save the augmented images
        utils.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


# def predict_images(test_data_path, print_speed=False):
#     num_classes = 2
#     image_shape = (160, 576)
#     runs_dir = './runs'

#     with tf.Session() as sess:
#         # Predict the logits
#         #input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
#         #nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
#         #logits = tf.reshape(nn_last_layer, (-1, num_classes))


#         # Restore the saved model
#         saver = tf.train.Saver()
#         saver.restore(sess, model_path)
#         print("Restored the saved Model in file: %s" % model_path)

#         # Predict the samples
#         utils.pred_samples(runs_dir, test_data_path, sess, image_shape, logits, keep_prob, input_image, print_speed)



if __name__ == '__main__':
    run()

         
         
         
   

    

       
  
