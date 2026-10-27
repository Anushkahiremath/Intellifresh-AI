DISCLAIMER = (
    "AI predictions are intended for educational and preliminary quality-assessment "
    "purposes only. Final consumption decisions should include physical inspection "
    "and proper food-safety practices."
)

def get_recommendation(prediction: str) -> dict:
    if prediction.lower() == "fresh":
        return {"summary": "The uploaded sample appears fresh according to the AI model.", "actions": [
            "Store under appropriate conditions.", "Avoid excessive moisture.",
            "Inspect physically before consumption.", "Consume within the appropriate storage period.",
        ]}
    return {"summary": "The uploaded sample shows visual characteristics associated with staleness.", "actions": [
        "Inspect the fruit or vegetable carefully.",
        "Avoid consumption if mold, unusual smell, leakage, or severe discoloration is present.",
        "Separate spoiled produce from fresh produce.", "Dispose of unsuitable produce safely.",
    ]}
