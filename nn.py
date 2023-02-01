from telegram import (
    
    Update,
    Message
    )
from telegram.constants import ChatAction
from telegram.ext import CallbackContext, ContextTypes


import os
import numpy as np
import pickle
from numpy.linalg import norm
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.neighbors import NearestNeighbors

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

async def file_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, message: Message = None):
    message = message or update.effective_message
    if not message:
        return

    # wait_message = update.message.reply_text("⌛ Give me a sec...")
    # a = context.bot.send_chat_action(chat_id=update.message.chat_id, action=ChatAction.TYPING)

    # a = update.message.reply_text(update.message.effective_attachment[0].file_id+'.png')
    # filename = context.bot.send_photo(chat_id=update.message.chat_id,  photo=update.message.effective_attachment[0].file_id)

    print(type(update.message.effective_attachment[0].file_id))

    a = await context.bot.getFile(update.message.effective_attachment[0].file_id)

    filename = f'uploads/+{update.message.effective_attachment[0].file_id}+.jpg'
    await a.download_to_drive(filename)

    def model_picker(name):
        if (name == 'vgg16'):
            model = VGG16(weights='imagenet',
                        include_top=False,
                        input_shape=(224, 224, 3),
                        pooling='max')
        elif (name == 'vgg19'):
            model = VGG19(weights='imagenet',
                        include_top=False,
                        input_shape=(224, 224, 3),
                        pooling='max')
        elif (name == 'mobilenet'):
            model = MobileNet(weights='imagenet',
                            include_top=False,
                            input_shape=(224, 224, 3),
                            pooling='max',
                            depth_multiplier=1,
                            alpha=1)
        elif (name == 'inception'):
            model = InceptionV3(weights='imagenet',
                                include_top=False,
                                input_shape=(224, 224, 3),
                                pooling='max')
        elif (name == 'resnet'):
            model = ResNet50(weights='imagenet',
                            include_top=False,
                            input_shape=(224, 224, 3),
                            pooling='max')
        # elif (name == 'xception'):
        #     model = Xception(weights='imagenet',
        #                     include_top=False,
        #                     input_shape=(224, 224, 3),
        #                     pooling='max')
        else:
            print("Specified model not available")
        return model

    def extract_features(img_path, model):
        input_shape = (224, 224, 3)
        img = image.load_img(img_path,
                            target_size=(input_shape[0], input_shape[1]))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        features = model.predict(preprocessed_img)
        flattened_features = features.flatten()
        normalized_features = flattened_features / norm(flattened_features)
        return normalized_features

    def recommend(features,feature_list):
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)

        distances, indices = neighbors.kneighbors([features])

        return indices

    # Helper function to get the classname and filename
    def classname_filename(str):
        return str.split('/')[-2] + '/' + str.split('/')[-1]

    filenames = pickle.load(open('filenames-caltech101.pickle', 'rb'))
    feature_list = pickle.load(open('features-caltech101-resnet.pickle',
                                'rb'))
    model_architecture = 'resnet'
    model = model_picker(model_architecture)
    features = extract_features(filename,model)
    indices = recommend(features,feature_list)

    res0 = update.message.reply_photo('dataset/'+ classname_filename(filenames[indices[0][0]]))
    res1 = update.message.reply_photo('dataset/'+ classname_filename(filenames[indices[0][1]]))
    res2 = update.message.reply_photo('dataset/'+ classname_filename(filenames[indices[0][2]]))
    res3 = update.message.reply_photo('dataset/'+ classname_filename(filenames[indices[0][3]]))


    print('dataset/'+classname_filename(filenames[indices[0][0]]))
    print('dataset/'+classname_filename(filenames[indices[0][1]]))

    await res0
    await res1
    await res2
    await res3
