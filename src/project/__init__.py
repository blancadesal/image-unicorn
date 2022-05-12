import os

from flask import Flask

from project.config import config


# instantiate the extensions
# ...


def create_app(config_name=None):

    if config_name is None:
        config_name = os.environ.get('FLASK_CONFIG', 'development')

    # instantiate the app
    app = Flask(__name__)

    # set config
    app.config.from_object(config[config_name])

    # set up extensions
    # ...

    # register blueprints
    # ...

    # shell context for flask cli
    @app.shell_context_processor
    def ctx():
        return {"app": app}

    return app