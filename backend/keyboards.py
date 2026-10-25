from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


def confirmation_keyboard() -> InlineKeyboardMarkup:
    """Кнопки подтверждения или отмены сохранения задачи.

    В bot.py должны быть обработчики callback_data:
    ``confirm_task`` и ``cancel_task``.
    """
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Сохранить", callback_data="confirm_task")
    builder.button(text="❌ Отмена", callback_data="cancel_task")
    builder.adjust(2)
    return builder.as_markup()
