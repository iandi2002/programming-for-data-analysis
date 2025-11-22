import json
import os

FILE_NAME = "history.json"



def save_history(text, label, prob):
    entry = {"text": text, "label": label, "prob": prob}

    try:
        if os.path.exists(FILE_NAME):
            with open(FILE_NAME, "r", encoding="utf8") as f:
                history = json.load(f)
        else:
            history = []
    except json.JSONDecodeError:
        history = []

    history.append(entry)

    with open(FILE_NAME, "w", encoding="utf8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("Сохранено в историю.")



def load_history():
    if not os.path.exists(FILE_NAME):
        print("История пуста.")
        return

    try:
        with open(FILE_NAME, "r", encoding="utf8") as f:
            history = json.load(f)
    except json.JSONDecodeError:
        print("Ошибка чтения файла истории.")
        return

    print("\n========== История ==========")
    for i, h in enumerate(history[-10:], 1):
        print(f"{i}. {h['text']} → {h['label']} ({h['prob']:.3f})")
