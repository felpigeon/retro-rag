from flask import Flask
from flask_cors import CORS
from flasgger import Swagger

from src.routes.rerank import rerank_bp
from src.routes.summarize import summarize_bp
from src.routes.detect_hallucination import detect_hallucination_bp


app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

app.register_blueprint(rerank_bp)
app.register_blueprint(summarize_bp)
app.register_blueprint(detect_hallucination_bp)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)