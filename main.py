from ml_pipeline import train_model, predict_comment
from storage import save_history, load_history

def menu():
    while True:
        print("\n========== Toxic Comment Detection ==========")
        print("1 — Обучить модель")
        print("2 — Предсказать токсичность текста")
        print("3 — Показать историю классификаций")
        print("4 — Выход")

        choice = input("Выберите действие: ")

        if choice == "1":
            train_model()
        elif choice == "2":
            text = input("Введите текст комментария: ")
            label, prob = predict_comment(text)
            print(f"\nРезультат: {label} (prob={prob:.4f})")
            save_history(text, label, prob)
        elif choice == "3":
            load_history()
        elif choice == "4":
            print("Выход.")
            break
        else:
            print("Ошибка: выберите пункт 1-4.")

if __name__ == "__main__":
    menu()
