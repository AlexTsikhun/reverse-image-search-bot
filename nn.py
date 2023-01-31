from telegram import (
    
    Update,
    Message
    )
from telegram.constants import ChatAction
from telegram.ext import CallbackContext, ContextTypes


async def file_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, message: Message = None):
    message = message or update.effective_message
    if not message:
        return

    # wait_message = update.message.reply_text("⌛ Give me a sec...")
    # a = context.bot.send_chat_action(chat_id=update.message.chat_id, action=ChatAction.TYPING)

    # a = update.message.reply_text(update.message.effective_attachment[0].file_id+'.png')
    filename = context.bot.send_photo(chat_id=update.message.chat_id,  photo=update.message.effective_attachment[0].file_id)

    print(filename)
    await filename