import asyncio


from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import DeleteWebhook
from aiogram.types import Message
from loguru import logger

from config import BOT_TOKEN, setup_logger
from model import QuestionAnsweringModel

model_name = "sergeyzh/LaBSE-ru-turbo"
state_dict_path = "best_base_model.pt"
qa_model = QuestionAnsweringModel(model_name, state_dict_path)


async def send_welcome(message: Message):
    await message.reply("Привет! Задай свой вопрос:")


async def answer_question(message: Message):
    question = message.text
    answer = qa_model.get_answer(question)
    await message.reply(answer)


async def main():
    bot = Bot(token=BOT_TOKEN)
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.message.register(send_welcome, CommandStart())
    dp.message.register(answer_question)
    await bot(DeleteWebhook(drop_pending_updates=True))
    logger.info("Запуск бота")
    await dp.start_polling(bot)


if __name__ == "__main__":
    setup_logger()
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.error("Бот остановлен!")
