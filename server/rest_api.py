from flask import Flask, request

from server.sentiment import Sentiment
from server.response_generator import ResponseGenerator

app = Flask(__name__)
chatbot: ResponseGenerator = None

IP_ADDRESS = "0.0.0.0"
PORT = 5000

INVALID_REQUEST = {"error": "Invalid request format"}, 400
INTERNAL_ERROR = {"error": "Internal server error, please try again"}, 500


def start_api(args, model, tokenizer):
    global chatbot, app, IP_ADDRESS, PORT
    chatbot = ResponseGenerator(args, model, tokenizer)

    from waitress import serve
    print(f"Server started on http://{IP_ADDRESS}:{PORT}")
    print("Press CTRL+C to stop the app")
    serve(app, host=IP_ADDRESS, port=PORT)



@app.post("/response")
def get_response():
    global chatbot, INTERNAL_ERROR, INVALID_REQUEST

    if not request.is_json:
        return INVALID_REQUEST

    input_request = request.get_json()

    if "utterance" not in input_request or "sentiment" not in input_request:
        return INVALID_REQUEST

    utterance = input_request["utterance"]
    sentiment = Sentiment.from_string(input_request["sentiment"])

    try:
        response = chatbot.get_response(sentiment, utterance)
    except Exception as e:
        print(e)
        return INTERNAL_ERROR

    return {"response": response}, 200


@app.get("/test")
def test():
    return {"OK": "ok"}
