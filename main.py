from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
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



app = ApplicationBuilder().token("6138462545:AAFp5SWIBzJlkXu1Vw_RAjBOnT3wWV-k7II").build()

app.add_handler(CommandHandler("hello", hello))
app.add_handler(MessageHandler(filters.PHOTO, file_handler))
# app.add_handler(MessageHandler(filters.ALL, send))



#MessageHandler(filters.PHOTO, photo)
app.run_polling()