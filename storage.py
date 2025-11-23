import json
import os

FILE_NAME = "history.json"

# Same label_map as in ml_pipeline, but defined here for pretty printing
label_map = {
    "IsToxic": "General toxicity",
    "IsAbusive": "Abusive / insulting language",
    "IsThreat": "Threats",
    "IsProvocative": "Provocative language",
    "IsObscene": "Obscene language",
    "IsHatespeech": "Hate speech",
    "IsRacist": "Racism",
    "IsNationalist": "Nationalism",
    "IsSexist": "Sexism",
    "IsHomophobic": "Homophobia",
    "IsReligiousHate": "Religious hate",
    "IsRadicalism": "Radicalism",
}


def save_history(text, label, prob, toxic_types=None):
    """
    Append a classification result to the history JSON file.

    toxic_types: list of (label_code, probability)
    """
    entry = {
        "text": text,
        "label": label,
        "prob": float(prob),
    }

    if toxic_types:
        entry["types"] = [
            {"code": code, "prob": float(p)} for code, p in toxic_types
        ]

    # Load existing history (if any)
    try:
        if os.path.exists(FILE_NAME):
            with open(FILE_NAME, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []
    except json.JSONDecodeError:
        history = []

    history.append(entry)

    # Save updated history
    with open(FILE_NAME, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_history():
    """
    Print the last 10 classification results from history.
    """
    if not os.path.exists(FILE_NAME):
        print("History is empty.")
        return

    try:
        with open(FILE_NAME, "r", encoding="utf-8") as f:
            history = json.load(f)
    except json.JSONDecodeError:
        print("Error reading history file.")
        return

    if not history:
        print("History is empty.")
        return

    print("\n========== History (last 10 entries) ==========")
    for i, h in enumerate(history[-10:], 1):
        text = h.get("text", "")
        label = h.get("label", "?")
        prob = h.get("prob", 0.0)
        print(f"{i}. {text} → {label} ({prob:.3f})")

        types = h.get("types") or []
        if types:
            pretty_types = []
            for t in types:
                code = t.get("code")
                p = t.get("prob", 0.0)
                name = label_map.get(code, code)
                pretty_types.append(f"{name} ({p:.2f})")

            if pretty_types:
                print("   Toxicity types: " + ", ".join(pretty_types))
