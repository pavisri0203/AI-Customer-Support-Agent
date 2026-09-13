from intents import INTENTS


def classify_intent(message):
    message = message.lower()

    scores = {}

    for intent, examples in INTENTS.items():
        score = 0

        for example in examples:
            words = example.lower().split()

            for word in words:
                if word in message:
                    score += 1

        scores[intent] = score

    best_intent = max(scores, key=scores.get)

    if scores[best_intent] == 0:
        return "general_issue"

    return best_intent