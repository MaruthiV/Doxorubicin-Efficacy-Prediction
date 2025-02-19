import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import nibabel as nib
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import segmentation_models_3D as sm
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

nifti_file = nib.load('/content/BraTS2021_00000_seg.nii.gz')
brain_numpy = np.asarray(nifti_file.dataobj) / np.max(brain_numpy)
nifti_file = nib.load('/content/BraTS2021_00000_seg.nii.gz')
brain_label = np.asarray(nifti_file.dataobj)
brain_label[brain_label == 4] = 3 # Make values between 0,1,2,3

x_train = []
y_train = []

from scipy.ndimage import zoom

test_cases = 0

for training_example in os.listdir('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/'):
  if test_cases > 20:
    break
  if len(os.listdir('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example)) == 0:
    continue

  prefix = os.listdir('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example)[0][0:15]

  flair_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_flair.nii.gz')
  flair_file = np.asarray(flair_file.dataobj)
  print(flair_file.shape)
  flair_file = zoom(flair_file, (0.4, 0.4, 0.61935483871)) / flair_file.max()

  t1_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_t1.nii.gz')
  t1_file = np.asarray(t1_file.dataobj)
  t1_file = zoom(t1_file, (0.4, 0.4, 0.61935483871)) / t1_file.max()

  t2_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_t2.nii.gz')
  t2_file = np.asarray(t2_file.dataobj)
  t2_file = zoom(t2_file, (0.4, 0.4, 0.61935483871)) / t2_file.max()

  #combined_arr = np.stack((flair_file, t1_file, t2_file), axis = 3)
  combined_arr = np.stack((flair_file, t1_file, t2_file), axis = 3)
  

  #x_train.append(np.array([[row] for row in combined_arr]))
  x_train.append(combined_arr)
  #x_train.append(flair_file)

  seg_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_seg.nii.gz')
  seg_file = np.asarray(seg_file.dataobj)
  
  
  seg_file[seg_file == 4] = 3 # Make values between 0,1,2,3
  seg_file = zoom(seg_file, (0.4, 0.4, 0.61935483871))
  seg_file[seg_file == 4] = 3

  y_train.append(seg_file)

  test_cases += 1

x_train = np.array(x_train)
y_train = np.array(y_train)

a = y_train[0]
b = np.zeros((96,96,96,4))

for x in range(len(y_train[0])):
  for y in range(len(y_train[0][0])):
    for z in range(len(y_train[0][0][0])):
      insert_arr = np.array([0,0,0,0])
      insert_arr[a[x][y][z]-1] = 1
      b[x,y,z] = np.array(insert_arr)

print (y_train[0].min())

print (b)

print (x_train[0].shape)
print (x_train[0].max())

x_test = []
y_test = []

test_cases = 0

for training_example in os.listdir('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/'):
  if test_cases > 120:
    break

  if test_cases < 100:
    test_cases += 1
    continue
  
  if len(os.listdir('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example)) == 0:
    continue

  prefix = os.listdir('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example)[0][0:15]

  flair_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_flair.nii.gz')
  flair_file = np.asarray(flair_file.dataobj)
  flair_file = zoom(flair_file, (0.4, 0.4, 0.61935483871)) / flair_file.max()

  t1_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_t1.nii.gz')
  t1_file = np.asarray(t1_file.dataobj)
  t1_file = zoom(t1_file, (0.4, 0.4, 0.61935483871)) / t1_file.max()

  t2_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_t2.nii.gz')
  t2_file = np.asarray(t2_file.dataobj)
  t2_file = zoom(t2_file, (0.4, 0.4, 0.61935483871)) / t2_file.max()

  #combined_arr = np.stack((flair_file, t1_file, t2_file), axis = 3)
  combined_arr = np.stack((flair_file, t1_file, t2_file), axis = 3)

  x_test.append(combined_arr)
  
  #x_test.append(flair_file)

  seg_file = nib.load('/content/drive/MyDrive/BRATS_2021_Data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/' + training_example + '/' + prefix + '_seg.nii.gz')
  seg_file = np.asarray(seg_file.dataobj)
  
  seg_file[seg_file == 4] = 3 # Make values between 0,1,2,3
  seg_file = zoom(seg_file, (0.4, 0.4, 0.61935483871))
  y_test.append(seg_file)

  test_cases += 1

x_test = np.array(x_test)
y_test = np.array(y_test)

model = sm.Unet(
            "seresnext101",
            input_shape = (96, 96, 96, 3),
            encoder_weights='imagenet',
            classes=4,
            activation='sigmoid'
        )

loss_to_use = sm.losses.bce_jaccard_loss
#model.compile(optimizer='adam', loss=loss_to_use, metrics=[sm.metrics.iou_score, sm.metrics.f1_score])
model.compile(optimizer='adam', loss=loss_to_use, metrics=[sm.metrics.iou_score, sm.metrics.f1_score])

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

history = model.fit(
    x_train,
    y_train,
    batch_size=1,
    epochs=20,
    validation_data=(x_test, y_test)
)

def calculate_dice_coefficient(pred, label):
  example_dice_coeff = np.sum([pred==label])
  example_dice_coeff = example_dice_coeff / (240 * 240 * 155)
  return example_dice_coeff

dice_coefficients = []


for x in range(len(y_test)):
  pred = model.predict(x_test[x:(x+1)])
  pred = np.argmax(pred, axis=4)
  label = y_test[x:(x+1)]
  dice_coefficients.append(calculate_dice_coefficient(pred, label))

#Average Dice Coefficient
sum(dice_coefficients) / len(dice_coefficients)

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

#Source: https://www.kaggle.com/code/jyotidabas/3d-mri-analysis

plot_model(model, 
           show_shapes = True,
           show_dtype=False,
           show_layer_names = True, 
           rankdir = 'TB', 
           expand_nested = False, 
           dpi = 70)


from ipywidgets import interact, interactive, IntSlider, ToggleButtons


#Source: https://www.kaggle.com/code/jyotidabas/3d-mri-analysis
def visualize_3d_labels(layer):
    mask = nib.load('./brain_images/BraTS2021_00284/BraTS2021_00284_seg.nii.gz').get_fdata()
    plt.imshow(mask[:,:,layer])
    plt.axis('off')
    plt.tight_layout()


interact(visualize_3d_labels, layer=(0, image_data.shape[2] - 1))

#To visualize, maybe multiply pred and label matrix by 255/3 and plot splices or 3D

inputs = Input((240, 240, 155,3))

model = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
#model = Dropout(0.1)(model)
model = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(model)

model = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(model)
#model = Dropout(0.1)(model)
model = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(model)

#model = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(model)
#model = Dropout(0.1)(model)
#model = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(model)

#model = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(model)
#model = Dropout(0.1)(model)
#model = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(model)

outputs = Conv3D(4, (1, 1, 1), activation='softmax', padding='same')(model)

model = Model(inputs=[inputs], outputs=[outputs])
model.summary()