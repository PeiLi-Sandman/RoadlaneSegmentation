import os.path
import tensorflow as tf
import warnings
from distutils.version import LooseVersion
import csv
import time
import utils
import numpy as np
import pandas as pd


#create path for the finally trained model 
final_path='./model/model.ckpt'

assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.' \
                                                            '  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check if use GPU or CPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



def load_vgg_model(sess, vgg_path):
    """
    This is a VGG16 model for this project.
    return: Tensors of VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
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



def add_nn_last(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    # Use a shorter variable name for simplicity
    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

    # Apply 1x1 convolution in place of fully connected layer
    fcn8 = tf.layers.conv2d(layer7, filters=num_classes, kernel_size=1, name="fcn8")

    # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that we can add skip connection with 4th layer
    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")

    # Add a skip connection between current final layer fcn8 and 4th layer
    fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

    # Upsample again
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

    # Add skip connection
    fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

    # Upsample again
    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=num_classes,
    kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")

    return fcn11



def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    # make 4d tensor to 2d tensor where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes),name="fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1,num_classes))
    # define loss function to calculate distance between label and prediction 
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label_reshaped[:])
    loss_op = tf.reduce_mean(cross_entropy_loss,name="fcn_loss")
    # define training operation find weight and 
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

    return logits, train_op, loss_op



def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
    
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
   

    with tf.Session() as sess:
        vgg_path = os.path.join(data_dir, 'vgg')
       
        get_batches_fn = utils.create_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        
        epochs = 30
        batch_size = 8
        

        # TF placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg_model(sess, vgg_path)

        nn_last_layer = add_nn_last(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
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
        save_path = saver.save(sess, final_path)
        print("Model is saved to file: %s" % save_path)

        # TODO: predict the testing data and save the augmented images
        utils.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)





if __name__ == '__main__':
    run()

         
         
         
   

    

       
  
