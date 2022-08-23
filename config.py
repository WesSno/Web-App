class Config(object):
    DEBUG = False
    TESTING = False

    CLIENT_UPLOAD = "app/static"
    CLIENT_DOWNLOAD = "static"
    ALLOWED_IMAGE_EXTENSIONS = ["CSV", "XLSX"]
    SECRET_KEY = "dLOpbaGVmUbbxmUPH9aWPw"


class ProductionConfig(Config):
    pass

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True

