from flask import Flask, jsonify, request
from clf import predictAlpha

app = Flask(__name__)

@app.route("/predictAlpha", methods=["POST"])
def predict_alpha():
  image = request.files.get("alpha")
  prediction = predictAlpha(image)
  return jsonify({"prediction": prediction}), 200

if __name__ == "__main__":
  app.run(debug=True)
