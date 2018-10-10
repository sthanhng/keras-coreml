import numpy as np
import os

# set the matplotlib backend so figures can be saved in the background
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


# ------------------------------------------------
# Parameters
# ------------------------------------------------

VALID_EXTS = (
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.tif',
    '.tiff'
)

IMAGE_DIMS = (96, 96, 3)

# -----------------------------------------------
# Helper functions
# -----------------------------------------------

def list_files(base_path, valid_exts=VALID_EXTS, contains=None):
    # loop over the directory
    for (root_dir, dir_names, file_names) in os.walk(base_path):
        # loop over the file names in the current directory
        for fn in file_names:
            if contains is not None and fn.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = fn[fn.rfind('.'):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(valid_exts):
                image_path = os.path.join(root_dir, fn).replace(' ', '\\ ')
                yield image_path


# list all images in a directory
def list_images(base_path, contains=None):
    return list_files(base_path, valid_exts=VALID_EXTS, contains=contains)


# plot the training loss and accuracy
def plot_loss_acc(model, epochs, save_path):
    plt.style.use('ggplot')
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), model.history['loss'], label='train_loss')
    plt.plot(np.arange(0, N), model.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, N), model.history['acc'], label='train_acc')
    plt.plot(np.arange(0, N), model.history['val_acc'], label='val_acc')
    plt.title('Training loss and accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='upper left')
    plt.savefig(save_path)
