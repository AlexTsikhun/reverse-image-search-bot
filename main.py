import os

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram.constants import ChatAction

from nn import *
from dotenv import load_dotenv

load_dotenv()


async def hello(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    context.bot.send_chat_action(
        chat_id=update.message.chat_id, action=ChatAction.TYPING
    )


async def send(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message_id = update.message.effective_attachment[0].file_id
    a = await context.bot.getFile(message_id)

    filename = f'uploads/+{message_id}+.jpg'
    await a.download_to_drive(filename)


async def button_handler(update, context):
    query = update.callback_query
    data = query.data

    if data == 'button1':
        await button(update, context)
    elif data == 'button2':
        def build_menu(buttons, n_cols, header_buttons=None, footer_buttons=None):
            menu = [buttons[i:i + n_cols] for i in range(0, len(buttons), n_cols)]
            if header_buttons:
                menu.insert(0, header_buttons)
            if footer_buttons:
                menu.append(footer_buttons)
            return menu

        button_list = [
            InlineKeyboardButton("Book 1", callback_data='1'),
            InlineKeyboardButton("Book 2", callback_data='2'),
            InlineKeyboardButton("Book 3", callback_data='3'),
            InlineKeyboardButton("Book 4", callback_data='4'),
            InlineKeyboardButton("Book 5", callback_data='5')
        ]
        reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=2))
        await context.bot.send_message(chat_id=update.effective_chat.id, text='Please select a book:', reply_markup=reply_markup)

    elif data == '1':
        data = int(data)
        button_open(data)
    elif data == '2':
        data = int(data)
        button_open(data)
    elif data == '3':
        data = int(data)
        button_open(data)
    elif data == '4':
        data = int(data)
        button_open(data)
    elif data == '5':
        data = int(data)
        button_open(data)


app = ApplicationBuilder().token(os.getenv("TOKEN")).build()

app.add_handler(CommandHandler("hello", hello))
app.add_handler(MessageHandler(filters.PHOTO, file_handler))

app.add_handler(CallbackQueryHandler(button_handler))

app.run_polling()
