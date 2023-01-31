from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.constants import ChatAction

from nn import *

async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.bot.send_chat_action(
                    chat_id=update.message.chat_id, action=ChatAction.TYPING)  
    
    # await update.message.reply_text(f'Hello {update.effective_user.first_name}')

async def send(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_photo(update.message) # list of photos



app = ApplicationBuilder().token("6138462545:AAFp5SWIBzJlkXu1Vw_RAjBOnT3wWV-k7II").build()

app.add_handler(CommandHandler("hello", hello))
app.add_handler(MessageHandler(filters.PHOTO, file_handler))


#MessageHandler(filters.PHOTO, photo)
app.run_polling()