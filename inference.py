import argparse
import pickle

from keras.preprocessing.image import img_to_array
from keras.models import load_model

from utils import *

#####################################################################
parse = argparse.ArgumentParser()
parse.add_argument('--model-name', required=True,
                   help='path to trained model model')
parse.add_argument('--label-bin', required=True,
                   help='path to label binarizer')
parse.add_argument('--sample', required=True,
                   help='path to input sample')
args = parse.parse_args()


#####################################################################
# load the sample to test
image = cv2.imread(args.sample)
output = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = image.astype('float') / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


#####################################################################
# load the trained model and the label binarizer
print('==> Loading network...')
model = load_model(args.model_name)
lb = pickle.loads(open(args.label_bin, 'rb').read())

# classify the input image
print('==> Classifying image...')
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]

filename = args.sample[args.sample.rfind(os.path.sep) + 1:]
correct = 'correct' if filename.rfind(label) != -1 else 'incorrect'

# build the label and draw the label on the image
text = '{}: {:.2f}% ({})'.format(label, proba[idx] * 100, correct)
output = resize(output, width=400)
cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, COLOR_YELLOW, 1)


#####################################################################
# show the output image
print('==> {}'.format(label))
cv2.imshow('Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('************************* Well Done ***************************')
