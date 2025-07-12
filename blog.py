from flask import Flask, render_template, request
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import InferenceClient
import os

client = InferenceClient(api_key=os.getenv("HF_TOKEN"))


app = Flask(__name__)

def generate_blog(paragraph_topic):
    response = client.chat.completions.create(
        model="HuggingFaceH4/zephyr-7b-beta",
        messages=[
            {
                "role": "user",
                "content": f"Write a paragraph about the following topic: {paragraph_topic}"
            }
        ],
    )
    return response.choices[0].message['content']

@app.route("/", methods=["GET", "POST"])
def index():
    paragraph = ""
    if request.method == "POST":
        topic = request.form["topic"]
        paragraph = generate_blog(topic)
    return render_template("index.html", paragraph=paragraph)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
