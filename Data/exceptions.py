class DataLoadError(Exception):
    def __init__(self, err_msg):
        Exception.__init__(self, err_msg)


class DataPreporcessingError(Exception):
    def __init__(self, err_msg):
        Exception.__init__(self, err_msg)