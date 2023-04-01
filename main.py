from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram.constants import ChatAction

from nn import *

async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.bot.send_chat_action(
                    chat_id=update.message.chat_id, action=ChatAction.TYPING)  
    
    # await update.message.reply_text(f'Hello {update.effective_user.first_name}')

async def send(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # await update.message.reply_photo() # list of photos
    a = await context.bot.getFile(update.message.effective_attachment[0].file_id)

    filename = f'uploads/+{update.message.effective_attachment[0].file_id}+.jpg'
    await a.download_to_drive(filename)
    # await update.message.reply_text('!')


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
        
        # callback_data named in show list: 1 - most similar, 5 - less similar 
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
        print("!", type(data), data)
        data = int(data)
        print("!!", type(data), data)
        button_open(data)
    elif data == '2':
        print("!", type(data), data)
        data = int(data)
        print("!!", type(data), data)
        button_open(data)
    elif data == '3':
        print("!", type(data), data)
        data = int(data)
        print("!!", type(data), data)
        button_open(data)
    elif data == '4':
        print("!", type(data), data)
        data = int(data)
        print("!!", type(data), data)
        button_open(data)
    elif data == '5':
        print("!", type(data), data)
        data = int(data)
        print("!!", type(data), data)
        button_open(data)


app = ApplicationBuilder().token("6138462545:AAFp5SWIBzJlkXu1Vw_RAjBOnT3wWV-k7II").build()

app.add_handler(CommandHandler("hello", hello))
app.add_handler(MessageHandler(filters.PHOTO, file_handler))
# app.add_handler(MessageHandler(filters.ALL, send))

app.add_handler(CallbackQueryHandler(button_handler))

#MessageHandler(filters.PHOTO, photo)
app.run_polling()