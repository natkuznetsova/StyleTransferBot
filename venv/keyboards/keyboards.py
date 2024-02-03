from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils.keyboard import ReplyKeyboardBuilder

from lexikon import LEXICON_RU

button_style1 = KeyboardButton(text=LEXICON_RU['style1'])
button_style2 = KeyboardButton(text=LEXICON_RU['style2'])
button_reject = KeyboardButton(text=LEXICON_RU['reject'])

main_buttons_builder = ReplyKeyboardBuilder()
main_buttons_builder.row(button_style1, button_style2, button_reject, width=3)

main_buttons: ReplyKeyboardMarkup = main_buttons_builder.as_markup(
    one_time_keyboard=True,
    resize_keyboard=True
)

