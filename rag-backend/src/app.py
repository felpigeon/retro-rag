from flask import Flask
from flask_cors import CORS
from flasgger import Swagger

from src.routes.ingest import ingest_bp
from src.routes.ask import ask_bp


app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

app.register_blueprint(ingest_bp)
app.register_blueprint(ask_bp)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 