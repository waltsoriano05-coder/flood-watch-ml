from flask import Flask, jsonify
import flood_watch_ml

app = Flask(__name__)


@app.route("/")
def home():
    return "Flood Watch ML Backend is running!"


@app.route("/predict")
def predict():
    try:
        result = flood_watch_ml.run_prediction()

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)