from aiogram import Router, types
from aiogram.types import Message
#from aiogram.client import bot
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import default_state, State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
import numpy as np
from keyboards import main_buttons
from lexikon import LEXICON_RU
from aiogram import F
from models import resize, denormalisation
from models import model_gan
from models import style_model
from PIL import Image
import os


router = Router()

path = os.getcwd()
res_path = os.path.join(path, 'data').replace(os.sep, '/')

class FSMStyle(StatesGroup):
    #choose_style = State()        # Состояние ожидания выбора стиля
    style1_ph1 = State()         # Выбран стиль 1, ожидание загрузки 1-го фото
    style1_ph2 = State()         #Выбран стиль 2, ожидание загрузки 2-го фото
    style2 = State()      # Выбран стиль 2, ожидание загрузки фото


# Этот хэндлер будет срабатывать на команду "/start"
@router.message(CommandStart(), StateFilter(default_state))
async def process_start_command(message: Message):
    await message.answer(text=LEXICON_RU['/start'], reply_markup=main_buttons)


# Этот хэндлер будет срабатывать на команду "/help"
@router.message(Command(commands='help'), StateFilter(default_state))
async def process_help_command(message: Message):
    await message.answer(text=LEXICON_RU['/help'], reply_markup=main_buttons)


def transfer(content_img, style_img=None):

    if style_img:
        content_img = resize(content_img, 's1')
        style_img = resize(style_img)
        final_img = denormalisation(style_model.transform(content_img, style_img), 's1').detach().numpy().swapaxes(0,2).swapaxes(0, 1) * 255
        final_img = final_img.astype(np.uint8)
        final_img = Image.fromarray(final_img)
    else:
        content_img = resize(content_img, 's2')
        final_img = denormalisation(model_gan(content_img), 's2').detach().numpy().swapaxes(0,2).swapaxes(0, 1) * 255
        final_img = final_img.astype(np.uint8)
        final_img = Image.fromarray(final_img)
    return final_img


user_dict = {}


#Этот хэндлер будет реагировать на нажатие любой кнопки
@router.message(F.text.in_([LEXICON_RU['style1'], LEXICON_RU['style2'], LEXICON_RU['reject']]), StateFilter(default_state))
async def process_style_button(message: Message, state: FSMContext):
    user_dict[message.chat.id] = dict(style_flag=None, photo1=None, photo2=None)
    user_dict[message.chat.id]['style_flag'] = message.text
    if message.text == LEXICON_RU['style1']:
        await message.answer('Загрузите 2 фото:\n1 - для контекста\n2 - для стиля')
        await state.set_state(FSMStyle.style1_ph1)
    elif message.text == LEXICON_RU['style2']:
        await message.answer('Загрузите фото для преобразования')
        await state.set_state(FSMStyle.style2)
    else:
        await message.answer('До новых встреч!')
        await state.clear()



#Этот хэндлер будет реагировать на загруженное фото
@router.message(StateFilter(FSMStyle.style1_ph1), F.photo)
async def photo_catcher(message: Message, state: FSMContext):
    image = message.photo[-1]
    file_info = await message.bot.get_file(image.file_id)
    user_dict[message.chat.id]['photo1'] = await message.bot.download_file(file_info.file_path)
    await message.answer('Отлично! Теперь загрузите еще одно фото!')
    await state.set_state(FSMStyle.style1_ph2)


@router.message(StateFilter(FSMStyle.style1_ph2), F.photo)
async def style1_transformer(message: Message, state: FSMContext):
    image = message.photo[-1]
    file_info = await message.bot.get_file(image.file_id)
    user_dict[message.chat.id]['photo2'] = await message.bot.download_file(file_info.file_path)
    await message.answer('Изображение генерируется...')
    final = transfer(user_dict[message.chat.id]['photo1'], user_dict[message.chat.id]['photo2'])
    final.save(os.path.join(res_path, f'{message.chat.id}final.jpg').replace(os.sep, '/'))
    user_dict[message.chat.id]['style_flag'] = None
    user_dict[message.chat.id]['photos'] = 0
    await message.answer_photo(types.FSInputFile(path=os.path.join(res_path, f'{message.chat.id}final.jpg').replace(os.sep, '/')),
                               caption="Ваше фото с перенесенным стилем!")
    await state.clear()



@router.message(StateFilter(FSMStyle.style2), F.photo)
async def style2_transformer(message: Message, state: FSMContext):
    image = message.photo[-1]
    file_info = await bot.get_file(image.file_id)
    user_dict[message.chat.id]['photo1'] = await bot.download_file(file_info.file_path)
    final = transfer(user_dict[message.chat.id]['photo1'])
    final.save(os.path.join(res_path, f'{message.chat.id}final.jpg').replace(os.sep, '/'))
    await message.answer_photo(types.FSInputFile(path=os.path.join(res_path, f'{message.chat.id}final.jpg').replace(os.sep, '/')),
                               caption="Да... Фрида уже не та!")
    await state.clear()

# Этот хэндлер будет срабатывать на любые сообщения,
# кроме команд "/start" и "/help"
@router.message()
async def send_echo(message: Message):
    await message.reply(text='Неизвестная команда!')
