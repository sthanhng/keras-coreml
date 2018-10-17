import argparse

from keras.preprocessing.image import img_to_array
from keras.models import load_model

from utils import *

#####################################################################
parse = argparse.ArgumentParser()
parse.add_argument('--model-name', required=True,
                   help='path to trained model model')
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
# load the trained model
print('==> Loading network...')
model = load_model(args.model_name)

# classify the input image
print('==> Classifying image...')
confidence = model.predict(image)[0]
idx = np.argmax(confidence)

classes = ['cat', 'dog']
label = classes[idx]

filename = args.sample[args.sample.rfind(os.path.sep) + 1:]

# build the label and draw the label on the image
text = '{}: {:.2f}%'.format(label, confidence[idx] * 100)
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
