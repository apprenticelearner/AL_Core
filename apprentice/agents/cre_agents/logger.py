import logging

class Logger():
    _instance = None

    def __new__(cls, *args):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'logger'):
            formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)s()] %(message)s")

            sh = logging.StreamHandler()
            sh.setLevel(logging.DEBUG)
            # formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
            sh.setFormatter(formatter)

            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(sh)

    # def log(self, message):
        # self.logger.debug(message)
