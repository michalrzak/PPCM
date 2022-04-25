from flask import Flask, request

from server.sentiment import Sentiment
from server.response_generator import ResponseGenerator

app = Flask(__name__)
chatbot: ResponseGenerator = None

invalid_request = {"error", "Invalid request format"}, 400
internal_error = {"error", "Internal server error, please try again"}, 500


def start_api(args, model, tokenizer):
    global chatbot, app
    chatbot = ResponseGenerator(args, model, tokenizer)
    app.run(debug=True, use_reloader=False)


@app.post("/response")
def get_response():
    global chatbot
    if not request.is_json:
        return invalid_request

    input_request = request.get_json()

    if "utterance" not in input_request or "sentiment" not in input_request:
        return invalid_request

    utterance = input_request["utterance"]
    sentiment = Sentiment.from_string(input_request["sentiment"])

    try:
        response = chatbot.get_response(sentiment, utterance)
    except Exception as e:
        return internal_error

    return {"response": response}, 200
