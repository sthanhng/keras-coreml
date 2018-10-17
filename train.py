import argparse
import random
import pickle
import cv2
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from VGGNet import VGGNet

from utils import *

#####################################################################
parse = argparse.ArgumentParser()
parse.add_argument('--data-dir', required=True,
                   help='path to input data')
parse.add_argument('--model-dir', type=str, default='saved-models',
                   help='path to output model')
parse.add_argument('--model-name', required=True,
                   help='the name of output model')
parse.add_argument('--num-epochs', type=int, default=100,
                   help='number of training epochs')
parse.add_argument('--batch-size', type=int, default=32,
                   help='batch size')
parse.add_argument('--lr', type=float, default=0.001,
                   help='learning rate for training')
parse.add_argument('--label-bin', required=True,
                   help='path to output label binarizer')
parse.add_argument('--plot', type=str, default='plot.png',
                   help="path to output accuracy/loss plot")
args = parse.parse_args()

#####################################################################
# print the arguments
print('----- info ------')
print('[i] Path of the training data: ', args.data_dir)
print('[i] Path of the trained model: ', args.model_dir)
print('[i] Name of the trained model: ', args.model_name)
print('[i] # training epochs: ', args.num_epochs)
print('[i] Training batch size: ', args.batch_size)
print('[i] Learning rate: ', args.lr)
print('[i] Path of the output label binarizer: ', args.label_bin)
print('[i] Path of the output acc/loss plot: ', args.plot)
print('########################################################\n')

data = []
labels = []

# check the directory to contain the trained model
if not os.path.exists(args.model_dir):
    print('==> Creating {} directory...'.format(args.model_dir))
    os.makedirs(args.model_dir)
else:
    print('==> Skipping create directory {}'.format(args.model_dir))


#####################################################################
# list the image paths and randomly shuffle them
print('==> Loading images...')
imagePaths = sorted(list(list_images(args.data_dir)))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label[:-1])

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.2,
                                                  random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True, fill_mode='nearest')

#####################################################################
# initialize the model
print('==> Compiling model...')
model = VGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                     depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(lr=args.lr, decay=args.lr / args.num_epochs)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

#####################################################################
# train the network
print('==> Training the network...')
history = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=args.batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // args.batch_size,
    epochs=args.num_epochs, verbose=1)

#####################################################################
# save the model
print('==> Saving the model...')
model.save(args.model_dir + args.model_name)

# save the label binarizer
print('==> Saving label binarizer...')
f = open(args.label_bin, 'wb')
f.write(pickle.dumps(lb))
f.close()

# plot the training loss and accuracy
plot_loss_acc(history, args.num_epochs, args.plot)
print('==> All done!')
print('**************************************************************')
