import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import nibabel as nib
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import cv2
import tensorflow_examples 
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
from google.colab import drive 
from google.colab.patches import cv2_imshow
from keras.models import load_model

#Access NifTi data from Drive 
drive.mount('/content/drive', force_remount=True)

#Populating training data array

TRAINING_DATA_SIZE = 100
testing_data = False

x_train = np.zeros(((TRAINING_DATA_SIZE)*155,128,128,3))
y_train = np.zeros(((TRAINING_DATA_SIZE)*155,128,128)) #240

start_index = 0

def add_new_training_examples(): #Need to increment start_index by TRAINING_DATA_SIZE every time

  global start_index
  
  test_cases = 0
  x_insert_index = 0
  y_insert_index = 0

  current_index = 0

  if start_index > 1000 and testing_data == False: #1640
    start_index = 0
  
  for training_example in os.listdir('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/'):
    if training_example == '.DS_Store':
      continue
    if test_cases >= TRAINING_DATA_SIZE:
      start_index += TRAINING_DATA_SIZE
      return
    if len(os.listdir('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example)) == 0:
      current_index += 1
      continue
    if current_index < start_index:
      current_index += 1
      continue
    if current_index >= (start_index + TRAINING_DATA_SIZE):
      start_index += TRAINING_DATA_SIZE
      return


    
    current_index += 1

    prefix = os.listdir('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example)[0][0:15]

    flair_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_flair.nii.gz')
    flair_file = np.asarray(flair_file.dataobj, dtype=np.float64)
    flair_file /= flair_file.max()
    flair_file *= 255

    t1_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_t1.nii.gz')
    t1_file = np.asarray(t1_file.dataobj, dtype=np.float64)
    t1_file /= t1_file.max()
    t1_file *= 255

    t2_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_t2.nii.gz')
    t2_file = np.asarray(t2_file.dataobj, dtype=np.float64)
    t2_file /= t2_file.max()
    t2_file *= 255


    #for splice in range(155):
    #  training_case = np.zeros((240,240,3))
    #  for x in range(240):
    #    for y in range(240):
    #      training_case[x,y,0] = flair_file[x,y,splice]
    #      training_case[x,y,1] = t1_file[x,y,splice]
    #      training_case[x,y,2] = t2_file[x,y,splice]
    #  x_train.append(training_case)
    combined_arr = np.stack((flair_file, t1_file, t2_file), axis = 3)

    #print(combined_arr.shape)
    #print(combined_arr[:,:,0,:])
    #print(f'inserting into {x_insert_index}')

    for splice in range(155):
      #x_train.append(combined_arr[:,:,splice,:])
      img_to_add = cv2.resize(combined_arr[:,:,splice,:], (128,128))
      x_train[x_insert_index] = img_to_add
      #print(f'inserting into {x_insert_index}')
      #x_train.append(img_to_add)
      x_insert_index += 1

    #combined_arr = np.stack((flair_file, t1_file, t2_file), axis = 3)
    

    #x_train.append(np.array([[row] for row in combined_arr]))
    #x_train.append(combined_arr)
    #x_train.append(flair_file)

    seg_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_seg.nii.gz')
    seg_file = np.asarray(seg_file.dataobj)
    
    
    seg_file[seg_file == 4] = 3 # Make values between 0,1,2,3


    for splice in range(155):
      #label = np.expand_dims(seg_file[:,:,splice], axis=2)
      #y_train.append(label)
      img_to_add = cv2.resize(seg_file[:,:,splice], (128,128))
      y_train[y_insert_index] = img_to_add
      #y_train.append(img_to_add)
      y_insert_index += 1

    test_cases += 1


add_new_training_examples()

#Populating training data array

TESTING_DATA_SIZE = 20

x_test = np.zeros(((TESTING_DATA_SIZE)*155,128,128,3))
y_test = np.zeros(((TESTING_DATA_SIZE)*155,128,128)) #240

start_index_test = 1000


  
test_cases = 0
x_insert_index = 0
y_insert_index = 0

current_index = 0

  
for training_example in os.listdir('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/'):
  if training_example == '.DS_Store':
    continue
  if test_cases >= TESTING_DATA_SIZE:
    break
  if len(os.listdir('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example)) == 0:
    current_index += 1
    continue
  if current_index < start_index_test:
    current_index += 1
    continue
  if current_index >= (start_index_test + TESTING_DATA_SIZE):
    break


    
  current_index += 1

  prefix = os.listdir('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example)[0][0:15]

  flair_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_flair.nii.gz')
  flair_file = np.asarray(flair_file.dataobj, dtype=np.float64)
  flair_file /= flair_file.max()
  flair_file *= 255

  t1_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_t1.nii.gz')
  t1_file = np.asarray(t1_file.dataobj, dtype=np.float64)
  t1_file /= t1_file.max()
  t1_file *= 255

  t2_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_t2.nii.gz')
  t2_file = np.asarray(t2_file.dataobj, dtype=np.float64)
  t2_file /= t2_file.max()
  t2_file *= 255


  combined_arr = np.stack((flair_file, t1_file, t2_file), axis = 3)


  for splice in range(155):
    img_to_add = cv2.resize(combined_arr[:,:,splice,:], (128,128))
    x_test[x_insert_index] = img_to_add
    x_insert_index += 1

  seg_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_seg.nii.gz')
  seg_file = np.asarray(seg_file.dataobj)
    
    
  seg_file[seg_file == 4] = 3 # Make values between 0,1,2,3


  for splice in range(155):
    img_to_add = cv2.resize(seg_file[:,:,splice], (128,128))
    y_test[y_insert_index] = img_to_add
    y_insert_index += 1

  test_cases += 1

x_train[0].shape
y_train[0].shape
y_train[100][100]

cv2_imshow(y_train[100] / 3 * 255)


def custom_loss_function(y_actual, y_pred):
  #print("testing")
  #print(y_actual[0])
  #print(y_pred[0])
  #y_actual = y_actual.numpy()
  #y_pred = y_pred.numpy().argmax(axis=2)
  dice_coefficients = []
  intersection_values = [0,0,0,0]
  combined_area_values = [0,0,0,0]
  for class_value in range(4):
      for x in range(128):
        for y in range(128):
          if y_pred[0,x,y,class_value] == 1 and y_actual[0,x,y] == class_value:
            intersection_values[class_value] += 1
            combined_area_values[class_value] += 2
          elif y_pred[0,x,y,class_value] == 1:
            combined_area_values[class_value] += 1
          elif y_actual[0,x,y] == class_value:
            combined_area_values[class_value] += 1
            #combined_area_values[class_value] += y_pred[x,y,z] == class_value + y_actual[x,y,z] == class_value

  return 2*intersection_values[1] / (combined_area_values[1]) + 2*intersection_values[2] / (combined_area_values[2]) + 2*intersection_values[3] / (combined_area_values[3])

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]


def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  seg_map = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  sep_conv_2D_1 = tf.keras.layers.SeparableConv2D(4, (3,3), padding='same')

  x = seg_map(x)
  #x = sep_conv_2D_1(x)
  #x = tf.math.argmax(x, axis=3)
  #x = sep_conv_2D_1(x)

  



  return tf.keras.Model(inputs=inputs, outputs=x)

OUTPUT_CLASSES = 4

model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.05),
              loss=custom_loss_function,
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #loss=tfa.losses.SigmoidFocalCrossEntropy(),
              metrics=['accuracy'])

tf.keras.utils.plot_model(model, show_shapes=True)

class TrainingDataCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    add_new_training_examples()

model = load_model('/content/trained_segmentation_model_final_1.h5')
model.summary()

tf.keras.utils.plot_model(model, show_shape=True)

history = model.fit(
    x_train,
    y_train,
    batch_size=20,
    epochs=20, #50
    #class_weights=class_weights
    #class_weight = {0:1.0, 1:500.0, 2:500.0, 3:500},
    #sample_weight_mode="temporal"
    #validation_freq=10,
    #callbacks=[TrainingDataCallback()],
    #validation_data=(x_test, y_test)
)

y_train[0].shape
model.save ('trained_segmentation_model_final_2.h5')

def show_example_prediction(index):
  example_pred = (model.predict(np.array([x_test[index]]))[0].argmax(axis=2) / 3 * 255).astype(int)
  #print(example_pred.shape)
  example_pred = np.array(example_pred, dtype=np.uint8)
  print("Model Prediction")
  cv2_imshow(example_pred)
  example_pred_map = cv2.applyColorMap(example_pred, cv2.COLORMAP_JET).astype(int)

  cv2_imshow(example_pred_map)

  print("Label")
  label = y_test[index] / 3 * 255
  label = np.array(label, dtype=np.uint8)
  cv2_imshow(label)
  label_map = cv2.applyColorMap(label, cv2.COLORMAP_JET).astype(int)
  
  cv2_imshow(label_map)
  

  #cv2.cvtColor(x_train[index,:,:,0], cv2.COLOR_GRAY2BGR).astype(int)
  
  print("Flair Overlay of Model Prediction")
  print(example_pred_map.shape)
  print(cv2.merge([x_test[index,:,:,0]*10,x_test[index,:,:,0]*10,x_test[index,:,:,0]*10]).shape)

  cv2_imshow(cv2.addWeighted(example_pred_map, 0.5, cv2.merge([x_test[index,:,:,0],x_test[index,:,:,0],x_test[index,:,:,0]]).astype(int), 0.5, 0))
  print("Flair Overlay of Label")
  cv2_imshow(cv2.addWeighted(label_map, 0.5, cv2.merge([x_test[index,:,:,0],x_test[index,:,:,0],x_test[index,:,:,0]]).astype(int), 0.5, 0))

  print("T1 Overlay of Model Prediction")
  cv2_imshow(cv2.addWeighted(example_pred_map, 0.5, cv2.merge([x_test[index,:,:,1],x_test[index,:,:,1],x_test[index,:,:,1]]).astype(int), 0.5, 0))
  print("T1 Overlay of Label")
  cv2_imshow(cv2.addWeighted(label_map, 0.5, cv2.merge([x_test[index,:,:,1],x_test[index,:,:,1],x_test[index,:,:,1]]).astype(int), 0.5, 0))

  print("T2 Overlay of Model Prediction")
  cv2_imshow(cv2.addWeighted(example_pred_map, 0.5, cv2.merge([x_test[index,:,:,2],x_test[index,:,:,2],x_test[index,:,:,2]]).astype(int), 0.5, 0))
  print("T2 Overlay of Label")
  cv2_imshow(cv2.addWeighted(label_map, 0.5, cv2.merge([x_test[index,:,:,2],x_test[index,:,:,2],x_test[index,:,:,2]]).astype(int), 0.5, 0))



show_example_prediction(100)

# 0 - Brain MRI - dark blue
# 1 - NCR (necrotic tumor parts) - Dead cancerous cells - aqua
# 2 - ED (peritumoral edematous/invaded tissue) - Swelled tissue around tumor - yellow
# 3 - ET (enhancing tumor) - red - actual tumor

#Model Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy vs. Epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
#Model Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss vs. Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')

model.summary()
show_example_prediction(0)
show_example_prediction(20) 
show_example_prediction(200)

#Iterate through every splice in testing data (1300-1600)
#Add area of overlap for every class to variables

#Add individual areas for every class to variables

# 2 * overlap / sum of both individual areas

#TODO: Make sure that you are using max value across dimension for output array since its one-hot encoded (arg-max)

dice_coefficients = []
volumes = []
volume_file_strings = []

current_index = 0
start_index = 1000
TRAINING_DATA_SIZE = 5 #TODO Change this
test_cases = 0
  
for training_example in os.listdir('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/'):
  if training_example == '.DS_Store':
    continue
  if test_cases >= TRAINING_DATA_SIZE:
    break
  if len(os.listdir('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example)) == 0:
    current_index += 1
    continue
  #print('Got here')
  if current_index < start_index:
    current_index += 1
    continue
  
  if current_index >= (start_index + TRAINING_DATA_SIZE):
    break

  

  current_index += 1
  
  prefix = os.listdir('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example)[0][0:15]

  flair_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_flair.nii.gz')
  flair_file = np.asarray(flair_file.dataobj, dtype=np.float64)
  flair_file /= flair_file.max()
  flair_file *= 255

  t1_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_t1.nii.gz')
  t1_file = np.asarray(t1_file.dataobj, dtype=np.float64)
  t1_file /= t1_file.max()
  t1_file *= 255

  t2_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_t2.nii.gz')
  t2_file = np.asarray(t2_file.dataobj, dtype=np.float64)
  t2_file /= t2_file.max()
  t2_file *= 255



  combined_arr = np.stack((flair_file, t1_file, t2_file), axis = 3)


  seg_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_seg.nii.gz')
  seg_file = np.asarray(seg_file.dataobj)
    
    
  seg_file[seg_file == 4] = 3

  dice_values = [0,0,0,0]

  intersection_values = [0,0,0,0]
  combined_area_values = [0,0,0,0]
  
  volume_file_strings.append(training_example)
  label_volume = 0
  pred_volume = 0

  for splice in range(155):
    
    img_to_add = cv2.resize(combined_arr[:,:,splice,:], (128,128))
    model_prediction = model.predict(np.array([img_to_add]))[0].argmax(axis=2).astype(int)

    label = cv2.resize(seg_file[:,:,splice], (128,128)).astype(int)
    seg_file[seg_file == 4] = 3


    for x in range(4):

        

      for prediction_pixel, label_pixel in zip(model_prediction.flatten(), label.flatten()):
        if prediction_pixel == x and label_pixel == x:
          intersection_values[x] += 1
        
        if x == 3 and prediction_pixel == 3:
          pred_volume += 1
        if x == 3 and label_pixel == 3:
          label_volume += 1


      #intersection_values[x] += np.sum((model_prediction == x) == (label == x))
      combined_area_values[x] += np.sum(model_prediction == x) + np.sum(label == x)

    
  
  for x in range(4):
    dice_values[x] = 2*intersection_values[x] / combined_area_values[x]
  dice_coefficients.append(dice_values)
  test_cases += 1
  volumes.append([label_volume, pred_volume])

final_dice_coefficients = [0,0,0,0]

for value in dice_coefficients:
  final_dice_coefficients[0] += value[0]
  final_dice_coefficients[1] += value[1]
  final_dice_coefficients[2] += value[2]
  final_dice_coefficients[3] += value[3]

final_dice_coefficients[0] /= len(dice_coefficients)
final_dice_coefficients[1] /= len(dice_coefficients)
final_dice_coefficients[2] /= len(dice_coefficients)
final_dice_coefficients[3] /= len(dice_coefficients)

final_dice_coefficients

volumes