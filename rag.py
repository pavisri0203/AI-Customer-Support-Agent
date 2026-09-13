import pandas as pd
from sentence_transformers import SentenceTransformer, util

df = pd.read_csv("sample.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_historical_reply(message):

    apple = df[df["author_id"] == "AppleSupport"]

    customer_messages = []
    replies = []

    for _, reply_row in apple.iterrows():

        customer_id = reply_row["in_response_to_tweet_id"]

        if pd.notna(customer_id):

            customer = df[
                df["tweet_id"].astype(str) == str(int(customer_id))
            ]

            if not customer.empty:

                customer_messages.append(
                    str(customer.iloc[0]["text"])
                )

                replies.append(
                    str(reply_row["text"])
                )

    if not customer_messages:
        return "No similar historical reply found."

    # Convert messages into embeddings
    message_embedding = model.encode(
        message,
        convert_to_tensor=True
    )

    historical_embeddings = model.encode(
        customer_messages,
        convert_to_tensor=True
    )

    # Find most similar historical conversation
    similarities = util.cos_sim(
        message_embedding,
        historical_embeddings
    )[0]

    best_index = similarities.argmax().item()
    best_score = similarities[best_index].item()

    # Confidence threshold
    if best_score < 0.35:
        return "No similar historical reply found."

    return replies[best_index]                
         


