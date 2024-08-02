
# In[1]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_mean(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) -                tf.reduce_mean((1-alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0))
    return focal_loss_fixed

def combined_loss(y_true, y_pred):
    fl = focal_loss()(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return fl + dl

def iou_metric(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.int64)
    y_pred = tf.cast(y_pred, tf.int64)
    
    intersection = tf.reduce_sum(tf.cast(tf.equal(y_pred, y_true), tf.float32))
    union = tf.reduce_sum(tf.cast(tf.logical_or(tf.cast(y_pred, tf.bool), tf.cast(y_true, tf.bool)), tf.float32))
    
    iou = intersection / (union + tf.keras.backend.epsilon())
    return iou

def DeepLabV3Plus(input_shape, num_classes):
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    

    for layer in base_model.layers[-50:]:
        layer.trainable = True

    input_a = base_model.get_layer('top_activation').output
    input_b = base_model.get_layer('block4a_expand_activation').output


    x = input_a
    x1 = layers.Conv2D(256, 1, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)
    x2 = layers.Conv2D(256, 3, padding='same', dilation_rate=6, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x3 = layers.Conv2D(256, 3, padding='same', dilation_rate=12, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x4 = layers.Conv2D(256, 3, padding='same', dilation_rate=18, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.Concatenate()([x1, x2, x3, x4])
    
    x = layers.Conv2D(256, 1, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x_shape = tf.keras.backend.int_shape(x)
    input_b_shape = tf.keras.backend.int_shape(input_b)
    
    x = layers.UpSampling2D(size=(input_b_shape[1] // x_shape[1], input_b_shape[2] // x_shape[2]), 
                            interpolation='bilinear')(x)
    
    x = layers.Concatenate()([x, input_b])
    x = layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.UpSampling2D(size=(input_shape[0] // input_b_shape[1], input_shape[1] // input_b_shape[2]), 
                            interpolation='bilinear')(x)
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(num_classes, 3, padding='same', activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    return model

def create_generator(img_path, mask_path, batch_size, target_size, num_classes, is_training=True):
    if is_training:
        image_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='reflect'
        )
        mask_datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='reflect'
        )
    else:
        image_datagen = ImageDataGenerator(rescale=1./255)
        mask_datagen = ImageDataGenerator()
    
    image_generator = image_datagen.flow_from_directory(
        img_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        seed=42
    )
    mask_generator = mask_datagen.flow_from_directory(
        mask_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        color_mode='grayscale',
        seed=42
    )
    while True:
        img = next(image_generator)
        mask = next(mask_generator)
        mask = tf.keras.utils.to_categorical(mask, num_classes=num_classes)
        yield img, mask

input_shape = (256, 256, 3)
num_classes = 4
batch_size = 8
epochs = 20
steps_per_epoch = 150
validation_steps = 45

model = DeepLabV3Plus(input_shape=input_shape, num_classes=num_classes)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss=combined_loss, 
              metrics=[iou_metric])

train_generator = create_generator(
    img_path='Downloads/seg/data_for_keras_aug/train_images/',
    mask_path='Downloads/seg/data_for_keras_aug/train_masks/',
    batch_size=batch_size,
    target_size=(256, 256),
    num_classes=num_classes,
    is_training=True
)
val_generator = create_generator(
    img_path='Downloads/seg/data_for_keras_aug/val_images/',
    mask_path='Downloads/seg/data_for_keras_aug/val_masks/',
    batch_size=batch_size,
    target_size=(256, 256),
    num_classes=num_classes,
    is_training=False
)

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}, Training Loss: {logs['loss']:.4f}, Training IoU: {logs['iou_metric']:.4f}, "
              f"Validation Loss: {logs['val_loss']:.4f}, Validation IoU: {logs['val_iou_metric']:.4f}")

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=[CustomCallback(), reduce_lr]
)

print("Training complete.")
model.summary()

print("\nTraining History:")
for i in range(len(history.history['loss'])):
    print(f"Epoch {i+1}, Training Loss: {history.history['loss'][i]:.4f}, "
          f"Training IoU: {history.history['iou_metric'][i]:.4f}, "
          f"Validation Loss: {history.history['val_loss'][i]:.4f}, "
          f"Validation IoU: {history.history['val_iou_metric'][i]:.4f}")


# In[17]:


additional_epochs = 5
print(f"Resuming training from epoch {epochs + 1}")
history_continued = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs + additional_epochs,  
    initial_epoch=epochs,  
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=[CustomCallback(), reduce_lr]
)

for key in history_continued.history:
    if key in history.history:
        history.history[key].extend(history_continued.history[key])
    else:
        history.history[key] = history_continued.history[key]

print("Additional training complete.")

print("\nAdditional Training History:")
for i in range(additional_epochs):
    epoch_num = epochs + i + 1
    print(f"Epoch {epoch_num}, ", end="")
    for metric in ['loss', 'iou_metric', 'val_loss', 'val_iou_metric']:
        if metric in history.history:
            value = history.history[metric][-additional_epochs + i]
            print(f"{metric.capitalize()}: {value:.4f}, ", end="")
    print()

plot_learning_curves(history)


# In[2]:


import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.metrics import MeanIoU


# In[18]:


val_generator = create_generator(
    img_path='Downloads/seg/data_for_keras_aug/val_images/',
    mask_path='Downloads/seg/data_for_keras_aug/val_masks/',
    batch_size=batch_size,
    target_size=(256, 256),
    num_classes=num_classes,
    is_training=False
)

color_map = np.array([
    [173, 255, 47],   # Class 0 - light green
    [34, 139, 34],    # Class 1 - dark green
    [210, 180, 140],  # Class 2 - light brown
    [173, 216, 230],   # Class 3 - blue
], dtype=np.uint8)

def apply_color_map(mask, color_map):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in range(len(color_map)):
        color_mask[mask == label] = color_map[label]
    return color_mask

def visualize_predictions(generator, model, num_batches=5, num_classes=4):
    for i in range(num_batches):
        test_image_batch, test_mask_batch = next(generator)
        test_mask_batch_argmax = np.argmax(test_mask_batch, axis=3)
        test_pred_batch = model.predict(test_image_batch)
        test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)
        iou_keras = MeanIoU(num_classes=num_classes)
        iou_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
        print(f"Batch {i+1} - Mean IoU =", iou_keras.result().numpy())
        
        for img_num in range(test_image_batch.shape[0]):
            plt.figure(figsize=(12, 8))
            plt.subplot(231)
            plt.title('Testing Image')
            plt.imshow(test_image_batch[img_num])
            plt.subplot(232)
            plt.title('Testing Label')
            plt.imshow(apply_color_map(test_mask_batch_argmax[img_num], color_map))
            plt.subplot(233)
            plt.title('Prediction on test image')
            plt.imshow(apply_color_map(test_pred_batch_argmax[img_num], color_map))
            plt.show()
visualize_predictions(val_generator, model, num_batches=3)


# In[19]:


from keras.preprocessing.image import load_img, img_to_array
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  
    return img, img_array

def apply_color_map(mask, color_map):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in range(len(color_map)):
        color_mask[mask == label] = color_map[label]
    return color_mask

def predict_single_image(image_array, model):
    pred_mask = model.predict(image_array)
    pred_mask_argmax = np.argmax(pred_mask, axis=-1)[0]
    return pred_mask_argmax

image_path = r'Downloads\seg\images\12.jpg'  
target_size = (256, 256)
original_img, img_array = load_and_preprocess_image(image_path, target_size=target_size)
predicted_mask = predict_single_image(img_array, model)

color_map = np.array([
    [173, 255, 47],   # Class 0 - light green
    [34, 139, 34],    # Class 1 - dark green
    [210, 180, 140],  # Class 2 - light brown
    [173, 216, 230],  # Class 3 - light blue
], dtype=np.uint8)

plt.figure(figsize=(12, 8))
plt.subplot(131)
plt.title('Original Image')
plt.imshow(original_img)
plt.subplot(132)
plt.title('Test Mask')
plt.imshow(apply_color_map(predicted_mask, color_map))
plt.show()


# In[28]:


import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
val_generator = create_generator(
    img_path='Downloads/seg/data_for_keras_aug/val_images/',
    mask_path='Downloads/seg/data_for_keras_aug/val_masks/',
    batch_size=batch_size,
    target_size=(256, 256),
    num_classes=num_classes,
    is_training=False
)
val_loss = []
val_iou = []
y_true_all = []
y_pred_all = []


for _ in range(validation_steps):
    x, y_true = next(val_generator)
    y_pred = model.predict(x)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    batch_val_loss = combined_loss(y_true, y_pred)
    val_loss.append(batch_val_loss)
    batch_val_iou = iou_metric(y_true, y_pred)
    val_iou.append(batch_val_iou)
    y_true_all.extend(np.argmax(y_true, axis=-1).flatten())
    y_pred_all.extend(np.argmax(y_pred, axis=-1).flatten())

avg_val_loss = np.mean(val_loss)
avg_val_iou = np.mean(val_iou)

val_accuracy = accuracy_score(y_true_all, y_pred_all)
val_precision = precision_score(y_true_all, y_pred_all, average='weighted')
val_recall = recall_score(y_true_all, y_pred_all, average='weighted')
val_f1 = f1_score(y_true_all, y_pred_all, average='weighted')
print(f'Validation Loss: {avg_val_loss:.4f}')
print(f'Validation IoU: {avg_val_iou:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')
print(f'Validation Precision: {val_precision:.4f}')
print(f'Validation Recall: {val_recall:.4f}')
print(f'Validation F1 Score: {val_f1:.4f}')


# In[29]:


from sklearn.metrics import confusion_matrix
import numpy as np
cm = confusion_matrix(y_true_all, y_pred_all)

class_accuracy = cm.diagonal() / cm.sum(axis=1)

print("Accuracy for each class:")
for i, accuracy in enumerate(class_accuracy):
    print(f"Class {i}: {accuracy:.4f}")

overall_accuracy = np.sum(cm.diagonal()) / np.sum(cm)
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")


# In[21]:


#validation
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
class_report = classification_report(y_true_all, y_pred_all)
print(class_report)


# In[22]:


import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

train_generator = create_generator(
    img_path='Downloads/seg/data_for_keras_aug/train_images/',
    mask_path='Downloads/seg/data_for_keras_aug/train_masks/',
    batch_size=batch_size,
    target_size=(256, 256),
    num_classes=num_classes,
    is_training=True
)
train_loss = []
train_iou = []
y_true_all = []
y_pred_all = []

for _ in range(steps_per_epoch):
    x, y_true = next(train_generator)
    y_pred = model.predict(x)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    batch_train_loss = combined_loss(y_true, y_pred)
    train_loss.append(batch_train_loss)
    
    batch_train_iou = iou_metric(y_true, y_pred)
    train_iou.append(batch_train_iou)
    y_true_all.extend(np.argmax(y_true, axis=-1).flatten())
    y_pred_all.extend(np.argmax(y_pred, axis=-1).flatten())

avg_train_loss = np.mean(train_loss)
avg_train_iou = np.mean(train_iou)

train_accuracy = accuracy_score(y_true_all, y_pred_all)
train_precision = precision_score(y_true_all, y_pred_all, average='weighted')
train_recall = recall_score(y_true_all, y_pred_all, average='weighted')
train_f1 = f1_score(y_true_all, y_pred_all, average='weighted')

print(f'Training Loss: {avg_train_loss:.4f}')
print(f'Training IoU: {avg_train_iou:.4f}')
print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Training Precision: {train_precision:.4f}')
print(f'Training Recall: {train_recall:.4f}')
print(f'Training F1 Score: {train_f1:.4f}')


# In[23]:


#training
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
class_report = classification_report(y_true_all, y_pred_all)
print(class_report)


# In[25]:


from sklearn.metrics import confusion_matrix
import numpy as np
cm = confusion_matrix(y_true_all, y_pred_all)

class_accuracy = cm.diagonal() / cm.sum(axis=1)

print("Accuracy for each class:")
for i, accuracy in enumerate(class_accuracy):
    print(f"Class {i}: {accuracy:.4f}")

overall_accuracy = np.sum(cm.diagonal()) / np.sum(cm)
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")


# In[27]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def visualize_predictions(model, val_generator, num_samples=5):
    for _ in range(num_samples):
        val_images, val_masks = next(val_generator)
        
        predictions = model.predict(val_images)
        
        val_masks = np.argmax(val_masks, axis=-1)
        pred_masks = np.argmax(predictions, axis=-1)
        
        for i in range(val_images.shape[0]):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original Image
            ax1.imshow(val_images[i])
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Validation Mask
            ax2.imshow(val_masks[i], cmap='gray')
            ax2.set_title('Validation Mask')
            ax2.axis('off')
            
            # Predicted Mask
            ax3.imshow(pred_masks[i], cmap='gray')
            ax3.set_title('Predicted Mask')
            ax3.axis('off')
            
            plt.tight_layout()
            plt.show()

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    'Downloads/seg/data_for_keras_aug/val_images/',
    target_size=(256, 256),
    batch_size=1,
    class_mode=None,
    seed=42
)

mask_datagen = ImageDataGenerator(rescale=1./255)
mask_generator = mask_datagen.flow_from_directory(
    'Downloads/seg/data_for_keras_aug/val_masks/',
    target_size=(256, 256),  
    batch_size=1,
    class_mode=None,
    color_mode='grayscale',
    seed=42
)

val_generator = zip(val_generator, mask_generator)
visualize_predictions(model, val_generator, num_samples=5)


# In[ ]:




