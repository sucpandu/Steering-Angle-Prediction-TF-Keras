

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau 
from keras import backend as K
from copy import deepcopy
from PIL import Image
import csv
import cv2
import random



def assemble_filelists(path='./data/driving_log.csv',
                       angle_adjust=0.15,
                       small_angle_keep=1,
                       keep_threshold=0.05):

    
    image_files_center = []
    image_files_left = []
    image_files_right = []
    steering_angles_center = []

    with open(path, newline='') as csvfile:
        csvread = csv.reader(csvfile, delimiter=',')
        for i in csvread:
            angle = float(i[3].strip())
            if abs(angle) <= keep_threshold:
                p = np.random.uniform(0,1)
                if small_angle_keep >= p:
                    steering_angles_center.append([angle, angle])
                    image_files_center.append(i[0].strip())
                    image_files_left.append(i[1].strip())
                    image_files_right.append(i[2].strip())
            else:
                steering_angles_center.append([angle, angle])
                image_files_center.append(i[0].strip())
                image_files_left.append(i[1].strip())
                image_files_right.append(i[2].strip())

    assert (len(steering_angles_center) == len(image_files_center) 
            == len(image_files_left) == len(image_files_right))

    image_files_center = np.array(image_files_center)
    image_files_left = np.array(image_files_left)
    image_files_right = np.array(image_files_right)
    steering_angles_center = np.array(steering_angles_center)



    steering_angles_left = np.copy(steering_angles_center)
    steering_angles_right = np.copy(steering_angles_center)
    steering_angles_left[:,0] += angle_adjust
    steering_angles_right[:,0] -= angle_adjust



    image_files = np.concatenate((image_files_center,
                                  image_files_left,
                                  image_files_right))

    steering_angles = np.concatenate((steering_angles_center,
                                      steering_angles_left,
                                      steering_angles_right))
    
    return image_files, steering_angles



def transform_incline(image, shift=(5,20), orientation='rand'):
    
    rows,cols,ch = image.shape
    
    hshift = np.random.randint(shift[0],shift[1]+1)
    vshift = hshift
    
    if orientation == 'rand':
        orientation = random.choice(['down', 'up'])
    
    if orientation == 'up':
        hshift = -hshift
        vshift = -vshift
    elif orientation != 'down':
        raise ValueError("No or unknown orientation given. Possible values are 'up' and 'down'.")
    
    pts1 = np.float32([[70,70],
                       [250,70],
                       [0,rows],
                       [cols,rows]])
    pts2 = np.float32([[70+hshift,70+vshift],
                       [250-hshift,70+vshift],
                       [0,rows],
                       [cols,rows]])
    

    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (cols, rows))

def transform_curvature(image, shift=(5,30), orientation='rand'):
    
    rows,cols,ch = image.shape
    
    shift = np.random.randint(shift[0],shift[1]+1)
    
    if orientation == 'rand':
        orientation = random.choice(['left', 'right'])
    
    if orientation == 'left':
        shift = -shift
    elif orientation != 'right':
        raise ValueError("No or unknown orientation given. Possible values are 'left' and 'right'.")
    
    pts1 = np.float32([[70,70],[250,70],[0,rows],[cols,rows]])
    pts2 = np.float32([[70+shift,70],[250+shift,70],[0,rows],[cols,rows]])
    

    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (cols, rows)), shift

def do_rotate(image, min=5, max=15, orientation='rand'):
    
    rows,cols,ch = image.shape
    
 
    random_rot = np.random.randint(min, max+1)
    
    if orientation == 'rand':
        rotation_angle = random.choice([-random_rot, random_rot])
    elif orientation == 'left':
        rotation_angle = random_rot
    elif orientation == 'right':
        rotation_angle = -random_rot
    else:
        raise ValueError("Orientation is optional and can only be 'left' or 'right'.")
    
    M = cv2.getRotationMatrix2D((cols/2,rows/2), rotation_angle, 1)
    return cv2.warpAffine(image, M, (cols, rows)), -rotation_angle

def do_translate(image, horizontal=(0,40), vertical=(0,10)):
    
    rows,cols,ch = image.shape
    
    x = np.random.randint(horizontal[0], horizontal[1]+1)
    y = np.random.randint(vertical[0], vertical[1]+1)
    x_shift = random.choice([-x, x])
    y_shift = random.choice([-y, y])
    
    M = np.float32([[1,0,x_shift],[0,1,y_shift]])
    return cv2.warpAffine(image, M, (cols, rows)), x_shift

def do_flip(image, orientation='horizontal'):
    if orientation == 'horizontal':
        return cv2.flip(image, 1)
    else:
        return cv2.flip(image, 0)

def do_scale(image, min=0.9, max=1.1):
    
    rows,cols,ch = image.shape
    

    scale = np.random.uniform(min, max)
    
    M = cv2.getRotationMatrix2D((cols/2,rows/2), 0, scale)
    return cv2.warpAffine(image, M, (cols, rows))

def change_brightness(image, min=0.5, max=2.0):
    
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    random_br = np.random.uniform(min,max)
    

    mask = hsv[:,:,2]*random_br > 255
    v_channel = np.where(mask, 255, hsv[:,:,2]*random_br)
    hsv[:,:,2] = v_channel
    
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

def histogram_eq(image):
    
    image1 = np.copy(image)
    
    image1[:,:,0] = cv2.equalizeHist(image1[:,:,0])
    image1[:,:,1] = cv2.equalizeHist(image1[:,:,1])
    image1[:,:,2] = cv2.equalizeHist(image1[:,:,2])
    
    return image1



def generate_batch(filenames,
                   labels,
                   batch_size=128,
                   resize=False,
                   gray=False,
                   equalize=False,
                   brightness=False,
                   flip=False,
                   incline=False,
                   curvature=False,
                   curve_correct=0.008,
                   rotate=False,
                   rot_correct=0.03,
                   translate=False,
                   trans_correct=0.003):

    
    assert len(filenames) == len(labels), "The lengths of `filenames` and `labels` must be equal."
    assert not (len(filenames) == 0 or filenames is None), "`filenames` cannot be empty."
    
    current = 0
    
    while True:
        
        batch_X, batch_y = [], []
        

        if current >= len(filenames):
            filenames, labels = shuffle(filenames, labels)
            current = 0
        
        for filename in filenames[current:current+batch_size]:
            with Image.open(filename) as img:

                batch_X.append(np.array(img))
        batch_y = deepcopy(labels[current:current+batch_size])
        current += batch_size
        

        
        if equalize:
            batch_X = [histogram_eq(img) for img in batch_X]
        
        if brightness:
            for i in range(len(batch_X)):
                p = np.random.uniform(0,1)
                if p >= (1-brightness[2]):
                    batch_X[i] = change_brightness(batch_X[i],
                                                   min=brightness[0],
                                                   max=brightness[1])

        if flip:
            if flip[1] == 0:
                for i in range(len(batch_X)):
                    p = np.random.uniform(0,1)
                    if p >= (1-flip[0]):
                        batch_X[i] = do_flip(batch_X[i])
                        batch_y[i,0:2] = -batch_y[i,0:2]
            elif flip[1] == 1 and flip[2] > 0:
                for i in range(len(batch_X)):
                    if batch_y[i] >= flip[2]:
                        p = np.random.uniform(0,1)
                        if p >= (1-flip[0]):
                            batch_X[i] = do_flip(batch_X[i])
                            batch_y[i,0:2] = -batch_y[i,0:2]
            elif flip[1] == 1 and flip[2] < 0:
                for i in range(len(batch_X)):
                    if batch_y[i] <= flip[2]:
                        p = np.random.uniform(0,1)
                        if p >= (1-flip[0]):
                            batch_X[i] = do_flip(batch_X[i])
                            batch_y[i,0:2] = -batch_y[i,0:2]
            elif flip[1] == 2:
                for i in range(len(batch_X)):
                    if abs(batch_y[i]) >= flip[2]:
                        p = np.random.uniform(0,1)
                        if p >= (1-flip[0]):
                            batch_X[i] = do_flip(batch_X[i])
                            batch_y[i,0:2] = -batch_y[i,0:2]
            elif flip[1] == 3:
                for i in range(len(batch_X)):
                    if abs(batch_y[i]) <= flip[2]:
                        p = np.random.uniform(0,1)
                        if p >= (1-flip[0]):
                            batch_X[i] = do_flip(batch_X[i])
                            batch_y[i,0:2] = -batch_y[i,0:2]
            else: raise ValueError("Unsupported mode passed: {}. Supported modes are 0, 1, 2, 3.".format(flip[1]))
        
        if incline:
            if incline[3] == 0:
                for i in range(len(batch_X)):
                    p = np.random.uniform(0,1)
                    if p >= (1-incline[2]):
                        batch_X[i] = transform_incline(batch_X[i], (incline[0], incline[1]))
            elif incline[3] == 1 and incline[4] > 0:
                for i in range(len(batch_X)):
                    if batch_y[i,1] >= incline[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-incline[2]):
                            batch_X[i] = transform_incline(batch_X[i], (incline[0], incline[1]))
            elif incline[3] == 1 and incline[4] < 0:
                for i in range(len(batch_X)):
                    if batch_y[i,1] <= incline[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-incline[2]):
                            batch_X[i] = transform_incline(batch_X[i], (incline[0], incline[1]))
            elif incline[3] == 2:
                for i in range(len(batch_X)):
                    if abs(batch_y[i,1]) >= incline[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-incline[2]):
                            batch_X[i] = transform_incline(batch_X[i], (incline[0], incline[1]))
            elif incline[3] == 3:
                for i in range(len(batch_X)):
                    if abs(batch_y[i,1]) <= incline[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-incline[2]):
                            batch_X[i] = transform_incline(batch_X[i], (incline[0], incline[1]))
            else: raise ValueError("Unsupported mode passed: {}. Supported modes are 0, 1, 2, 3.".format(incline[3]))
            
                    
        if curvature:
            if curvature[3] == 0:
                for i in range(len(batch_X)):
                    p = np.random.uniform(0,1)
                    if p >= (1-curvature[2]):
                        if batch_y[i,1] > 0:
                            batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                     (curvature[0], curvature[1]),
                                                                     orientation='right')
                            batch_y[i,0] += curve_correct * cshift
                        elif batch_y[i,1] < 0:
                            batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                     (curvature[0], curvature[1]),
                                                                     orientation='left')
                            batch_y[i,0] += curve_correct * cshift
                        else:
                            batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                     (curvature[0], curvature[1]),
                                                                     orientation='rand')
                            batch_y[i,0] += curve_correct * cshift
            elif curvature[3] == 1 and curvature[4] > 0:
                for i in range(len(batch_X)):
                    if batch_y[i,1] >= curvature[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-curvature[2]):
                            batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                     (curvature[0], curvature[1]),
                                                                     orientation='right')
                            batch_y[i,0] += curve_correct * cshift
            elif curvature[3] == 1 and curvature[4] < 0:
                for i in range(len(batch_X)):
                    if batch_y[i,1] <= curvature[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-curvature[2]):
                            batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                     (curvature[0], curvature[1]),
                                                                     orientation='right')
                            batch_y[i,0] += curve_correct * cshift
            elif curvature[3] == 2:
                for i in range(len(batch_X)):
                    if abs(batch_y[i,1]) >= curvature[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-curvature[2]):
                            if batch_y[i,1] > 0:
                                batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                         (curvature[0], curvature[1]),
                                                                         orientation='right')
                                batch_y[i,0] += curve_correct * cshift
                            elif batch_y[i,1] < 0:
                                batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                         (curvature[0], curvature[1]),
                                                                         orientation='left')
                                batch_y[i,0] += curve_correct * cshift
            elif curvature[3] == 3:
                for i in range(len(batch_X)):
                    if abs(batch_y[i,1]) <= curvature[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-curvature[2]):
                            if batch_y[i,1] > 0:
                                batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                         (curvature[0], curvature[1]),
                                                                         orientation='right')
                                batch_y[i,0] += curve_correct * cshift
                            elif batch_y[i,1] < 0:
                                batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                         (curvature[0], curvature[1]),
                                                                         orientation='left')
                                batch_y[i,0] += curve_correct * cshift
                            else:
                                batch_X[i], cshift = transform_curvature(batch_X[i],
                                                                         (curvature[0], curvature[1]),
                                                                         orientation='rand')
            else: raise ValueError("Unsupported mode passed: {}. Supported modes are 0, 1, 2, 3.".format(curvature[3]))
        
        if rotate:
            if rotate[3] == 0:
                for i in range(len(batch_X)):
                    p = np.random.uniform(0,1)
                    if p >= (1-rotate[2]):
                        batch_X[i], rshift = do_rotate(batch_X[i], rotate[0], rotate[1])
                        batch_y[i,0] += rot_correct * rshift
            elif rotate[3] == 1 and rotate[4] > 0:
                for i in range(len(batch_X)):
                    if batch_y[i,1] >= rotate[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-rotate[2]):
                            batch_X[i], rshift = do_rotate(batch_X[i], rotate[0], rotate[1])
                            batch_y[i,0] += rot_correct * rshift
            elif rotate[3] == 1 and rotate[4] < 0:
                for i in range(len(batch_X)):
                    if batch_y[i,1] <= rotate[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-rotate[2]):
                            batch_X[i], rshift = do_rotate(batch_X[i], rotate[0], rotate[1])
                            batch_y[i,0] += rot_correct * rshift
            elif rotate[3] == 2:
                for i in range(len(batch_X)):
                    if abs(batch_y[i,1]) >= rotate[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-rotate[2]):
                            batch_X[i], rshift = do_rotate(batch_X[i], rotate[0], rotate[1])
                            batch_y[i,0] += rot_correct * rshift
            elif rotate[3] == 3:
                for i in range(len(batch_X)):
                    if abs(batch_y[i,1]) <= rotate[4]:
                        p = np.random.uniform(0,1)
                        if p >= (1-rotate[2]):
                            batch_X[i], rshift = do_rotate(batch_X[i], rotate[0], rotate[1])
                            batch_y[i,0] += rot_correct * rshift
            else: raise ValueError("Unsupported mode passed: {}. Supported modes are 0, 1, 2, 3.".format(rotate[3]))
            
        if translate:
            for i in range(len(batch_X)):
                p = np.random.uniform(0,1)
                if p >= (1-translate[2]):
                    batch_X[i], hshift = do_translate(batch_X[i], translate[0], translate[1])
                    batch_y[i,0] += trans_correct * hshift
        
        if resize:
            batch_X = [cv2.resize(img, dsize=resize) for img in batch_X]
            
        if gray:
            batch_X = np.expand_dims(np.asarray([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                                                 for img in batch_X]), 3)
        
        yield (np.array(batch_X), np.array(batch_y[:,0]))



def build_model():
    

    in_row, in_col, ch = 80, 160, 3
    

    cr_lef, cr_rig, cr_top, cr_bot = 5, 5, 20, 10
    cr_row, cr_col = in_row-cr_top-cr_bot, in_col-cr_lef-cr_rig
    
    model = Sequential()
    
    model.add(Cropping2D(cropping=((cr_top, cr_bot), (cr_lef, cr_rig)),
                         input_shape=(in_row, in_col, ch)))
    model.add(Lambda(lambda x: x/127.5 - 1., 
                     output_shape=(cr_row, cr_col, ch)))
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode="valid"))
    model.add(BatchNormalization(axis=3, momentum=0.99))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(BatchNormalization(axis=3, momentum=0.99))
    model.add(ELU())
    model.add(Convolution2D(128, 3, 3, subsample=(2, 2), border_mode="valid"))
    model.add(BatchNormalization(axis=3, momentum=0.99))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(512))
    model.add(BatchNormalization(axis=1, momentum=0.99))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)
    
    model.compile(optimizer=adam, loss="mse")
    
    return model



image_files, steering_angles = assemble_filelists()


X_train, X_val, y_train, y_val = train_test_split(image_files,
                                                  steering_angles,
                                                  test_size=0.2)

batch_size = 128
epochs = 1

train_generator = generate_batch(X_train,
                                 y_train,
                                 batch_size=batch_size,
                                 resize=(160,80),
                                 brightness=(0.4, 1.1, 0.1),
                                 flip=(0.5, 0),
                                 curvature=(5, 30, 0.5, 0),
                                 curve_correct=0.008,
                                 translate=((0, 40), (0, 10), 0.5),
                                 trans_correct=0.003)

val_generator = generate_batch(X_val,
                               y_val,
                               batch_size=batch_size,
                               resize=(160,80))


K.clear_session()

model = build_model()




history = model.fit_generator(generator = train_generator,
                              samples_per_epoch = len(y_train),
                              nb_epoch = epochs,
                              callbacks = [EarlyStopping(monitor='val_loss',
                                                         min_delta=0.0001,
                                                         patience=2),
                                           ReduceLROnPlateau(monitor='val_loss',
                                                             factor=0.2,
                                                             patience=1,
                                                             epsilon=0.0001,
                                                             cooldown=0)],
                              validation_data = val_generator,
                              nb_val_samples = len(y_val))

model_name = 'model_16'
model.save('./{}.h5'.format(model_name))
model.save_weights('./{}_weights.h5'.format(model_name))

print()
print("Model saved as {}.h5".format(model_name))
print("Weights also saved separately as {}_weights.h5".format(model_name))
print()
