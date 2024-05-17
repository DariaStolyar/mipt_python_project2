import os

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

from assets.lexicon import START, HELP
from src.model import train_model
from src.parsing import get_mega_dataset
from src.replicas import *

stats = {'intent': 0, 'generative': 0, 'stub': 0}
clf, vectorizer = train_model()
mega_dataset = get_mega_dataset()


def bot(text: str) -> str:
    # NLU
    intent = get_intent(text, clf, vectorizer)

    # Answer generation

    # rules
    if intent:
        stats['intent'] += 1
        return response_by_intent(intent)

    # generative model
    replica = get_generative_replica(text, mega_dataset)
    if replica:
        stats['generative'] += 1
        return replica

    # return failure phrase
    stats['stub'] += 1
    return get_failure_phrase()


def start(update: Updater, context) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text(START)


def help_command(update: Updater, context) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text(HELP)


def ask_bot(update: Updater, context) -> None:
    response = bot(update.message.text)
    count = sum(stats.values())
    print(f'{count}: intents {stats["intent"] / count:.2f}% '
          f'generative {stats["generative"] / count:.2f}% '
          f'stub {stats["stub"] / count:.2f}%', update.message.text, response)
    update.message.reply_text(response)


def main() -> None:
    """Start the bot."""
    with open(os.path.join('assets', '.env')) as f:
        token = f.readline().strip()
    updater = Updater(token, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(MessageHandler(Filters.text, ask_bot))
    updater.start_polling()
    updater.idle()
