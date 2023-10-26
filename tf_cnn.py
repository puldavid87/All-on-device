import cnn_tf_models as cnn_tf 
import EfficientNetB0 as Efficient
import InceptionV3 as Inception
import MobileNetV2 as Mobilenet
import vgg16 as VGG16
import os
import tensorflow as tf

#variables
classes = 4
epochs = 1
unfreeze_layers = -20
img_height = 224
img_width = 224
# Define some parameters for the loader:
batch_size = 32
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

models = ["Inception", "Mobilenet"]
path_data_source = "your_path"
test_dir = "your_path"
train_data, validation_data, test_data = cnn_tf.split_tratin_test_set(path_data_source,batch_size,img_height, img_width)

data_augmentation = True

efficient_model = Efficient.build_model(classes, data_augmentation)
inceptionV3 = Inception.build_model(classes, data_augmentation)
mobilenet_model = Mobilenet.build_model(classes, data_augmentation) 

    #Models
ml_models = [inceptionV3,mobilenet_model]
for i, folder_name in enumerate (models):
    path_model_destination = "your_path" + folder_name + "/"
    cnn_tf.make_folder(folder_name, path_model_destination) 
    model, history = cnn_tf.train_model(ml_models[i], train_data, validation_data, test_data, 
                            callback, path_model_destination,epochs,name=str(folder_name) 
                            )
      
    cnn_tf.unfreeze_model(ml_models[i], unfreeze_layers )
    model, history  =cnn_tf.train_model(ml_models[i], train_data, validation_data, test_data, 
                            callback, path_model_destination,epochs,name=str(folder_name) 
                            + "_"  + "unfree")


        
    