import os
import cv2
import numpy as np
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from ultralytics import YOLO
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes


# загрузка переменных из .env
load_dotenv()

# получаем доступы из окружения
TOKEN = os.getenv("BOT_TOKEN")

# model = YOLO("yolov8n.pt")
model = YOLO("yolov8n-seg.pt")
history = {}

# команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    history[user_id] = []
    await update.message.reply_text("Привет! Пришли мне фотографию, я найду на ней объекты.")

# получение фото
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    photo = update.message.photo[-1]
    file = await photo.get_file()
    os.makedirs("photos", exist_ok=True)
    path = f"photos/{user_id}_latest.jpg"
    await file.download_to_drive(path)

    # детекция
    results = model(path)
    names = list(set([model.names[int(c)] for c in results[0].boxes.cls.tolist()]))

    # сохраняем в историю
    # history[user_id] = [(path, names)]
    history[user_id] = (path, names)

    # отвечаем
    if len(names):
        await update.message.reply_text(
            f"Нашёл на фото: {', '.join(names)}\n\nОтправь через пробел классы, которые хочешь выделить."
        )
    else:
        await update.message.reply_text(f"Не удалось найти ни один объект")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id not in history or not history[user_id]:
        await update.message.reply_text("Сначала пришли фото.")
        return

    selected = update.message.text.lower().split()
    path, detected = history[user_id]
    results = model(path)
    img = cv2.imread(path)

    # зелёная маска
    mask_color = (0, 255, 0)
    
    # прозрачность
    alpha = 0.4

    counts = {}

    for r in results:
        for seg, cls_id in zip(r.masks.xy, r.boxes.cls):
            cls_name = model.names[int(cls_id)]
            if cls_name.lower() in selected:
                # нарисовать маску
                polygon = np.array([seg], dtype=np.int32)
                overlay = img.copy()
                cv2.fillPoly(overlay, polygon, mask_color)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                counts[cls_name] = counts.get(cls_name, 0) + 1

    out_path = f"photos/{user_id}_seg.jpg"
    cv2.imwrite(out_path, img)

    await update.message.reply_photo(photo=open(out_path, "rb"))

    if counts:
        stats = ", ".join([f"{k} — {v}" for k, v in counts.items()])
        await update.message.reply_text(f"Обнаружено: {stats}")
    else:
        await update.message.reply_text("Не нашёл выбранные классы на фото.")

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling()

if __name__ == "__main__":
    main()