import coremltools
import argparse
import pickle
import os

from keras.models import load_model

#####################################################################
parse = argparse.ArgumentParser()
parse.add_argument('--model', required=True, help='path to the trained model')
parse.add_argument('--coreml-model', type=str, default='coreml-models/', help='path to the converted model')
parse.add_argument('--label-bin', required=True, help='path to label binarizer')
args = parse.parse_args()

if not os.path.exists(args.coreml_model):
    print('==> Creating {} directory...'.format(args.coreml_model))
    os.makedirs(args.coreml_model)
else:
    print('==> Skipping create directory {}'.format(args.coreml_model))

#####################################################################
# Load the class labels
print('==> Loading class labels from label binarizer...')
lb = pickle.loads(open(args.label_bin, 'rb').read())
class_labels = lb.classes_.tolist()
print('==> Class labels: {}'.format(class_labels))

#####################################################################
# Load the trained model
print('==> Loading the trained model...')
model = load_model(args.model)

# Convert the model to the Core ML model format
print('==> Converting the model to the Core ML...')
coreml_model = coremltools.converters.keras.convert(model,
                                                    input_names='image',
                                                    image_input_names='image',
                                                    image_scale=1 / 255.0,
                                                    class_labels=class_labels,
                                                    is_bgr=True)

# Save the converted model
output = args.model.rsplit('.', 1)[0] + '.mlmodel'
output = output.rsplit('/')[-1]
print('==> Saving model as {}'.format(output))
coreml_model.save(args.coreml_model + output)
