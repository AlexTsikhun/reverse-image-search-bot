import pickle

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from telegram import (
    InputMediaPhoto,
    Update,
    Message,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)
from telegram.ext import ContextTypes
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing import image

from browser import go_to_site


filenames = pickle.load(open('features-caltech101-resnet.pickle', 'rb', encoding="utf-8", errors="ignore"))


def model_picker(name):
    if (name == 'vgg16'):
        model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='max'
        )
    elif (name == 'vgg19'):
        model = VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='max'
        )
    elif (name == 'mobilenet'):
        model = MobileNet(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='max',
            depth_multiplier=1,
            alpha=1
        )
    elif (name == 'inception'):
        model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='max'
        )
    elif (name == 'resnet'):
        model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='max'
        )
    elif (name == 'xception'):
        model = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='max'
        )
    else:
        print("Specified model not available")
    return model


def extract_features(img_path, model):
    input_shape = (224, 224, 3)
    img = image.load_img(
        img_path,
        target_size=(input_shape[0], input_shape[1])
    )
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices, distances


def classname_filename(str):
    return str.split('/')[-2] + '/' + str.split('/')[-1]


async def file_handler(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        message: Message = None
):
    message = message or update.effective_message
    if not message:
        return

    a = await context.bot.getFile(update.message.effective_attachment[0].file_id)
    filename = f'uploads/+{update.message.effective_attachment[0].file_id}+.jpg'
    await a.download_to_drive(filename)

    my_path = 'D:/STUDY/4ND-YEAR/GradWork/image_search_bot/data/book-covers/'

    feature_list = pickle.load(open('features-book-covers-dataset-resnet.pickle', 'rb'))
    model_architecture = 'resnet'
    model = model_picker(model_architecture)
    features = extract_features(filename, model)
    global distances
    global indices
    indices, distances = recommend(features, feature_list)

    a = context.bot.sendMediaGroup(
        chat_id=update.message.chat_id, media=[
            InputMediaPhoto(open(my_path + classname_filename(filenames[indices[0][1]]), 'rb')),
            InputMediaPhoto(open(my_path + classname_filename(filenames[indices[0][2]]), 'rb')),
            InputMediaPhoto(open(my_path + classname_filename(filenames[indices[0][3]]), 'rb')),
            InputMediaPhoto(open(my_path + classname_filename(filenames[indices[0][4]]), 'rb')),
            InputMediaPhoto(open(my_path + classname_filename(filenames[indices[0][5]]), 'rb'))
        ]
    )

    await a

    button1 = InlineKeyboardButton(text='Distances', callback_data='button1')
    button2 = InlineKeyboardButton(text='Choose book', callback_data='button2')

    keyboard = [[button1, button2]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    ret = context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Click to show distances",
        reply_markup=reply_markup
    )
    await ret

    return indices


async def button(update, context):
    list_distances = [round(elem, 3) for elem in distances[0].tolist()]
    query = update.callback_query

    await context.bot.send_message(
        text=list_distances,
        chat_id=query.message.chat_id
    )


def button_open(book_number):
    dt = pd.read_csv("data/main_dataset.csv")

    book_name = (dt[dt['img_paths'] == "dataset/" + classname_filename(filenames[indices[0][book_number]])]["name"].
                 to_string())
    # Split on 2 chunk and use second one, maxsplit=1 - split on 2 part sting; maxsplit=2 - 3 chunk. n - n+1 chunk
    book_name = book_name.split(maxsplit=1)[1]
    go_to_site(book_name)
