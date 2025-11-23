from ml_pipeline import train_model, predict_comment, load_model, label_map
from storage import save_history, load_history


def menu():
    print("Trying to load a saved model...")
    load_model()

    while True:
        print("\n========== Toxic Comment Detection ==========")
        print("1 — Train / retrain model")
        print("2 — Predict toxicity of a comment")
        print("3 — Show classification history")
        print("4 — Exit")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            train_model()

        elif choice == "2":
            text = input("Enter a comment (English or Russian): ")
            label, prob, toxic_types = predict_comment(text, visualize=True)

            print(f"\nResult: {label} (overall confidence {prob:.4f})")

            if toxic_types:
                print("\nDetected toxicity types:")
                for code, p in toxic_types:
                    human_name = label_map.get(code, code)
                    print(f"- {human_name} (probability {p:.2f})")
            else:
                if label == "Toxic":
                    print(
                        "Model marked the comment as toxic, "
                        "but did not activate specific subtypes "
                        "(triggered mostly by general toxicity)."
                    )
                else:
                    print("No specific toxicity types detected.")

            # Save to history including toxicity types
            save_history(text, label, prob, toxic_types)

        elif choice == "3":
            load_history()

        elif choice == "4":
            print("Exiting. Bye!")
            break

        else:
            print("Invalid option. Please choose 1–4.")


if __name__ == "__main__":
    menu()
