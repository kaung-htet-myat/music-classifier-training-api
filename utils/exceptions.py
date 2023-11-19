class ParameterNotProvidedError(Exception):
    """ParameterNotProvidedError"""

    def __init__(self, *args):
        if args:
            message = args[0]
            super().__init__(message)
            self.message = message


class UnsupportedParameterError(Exception):
    """UnsupportedParameterError"""

    def __init__(self, *args):
        if args:
            message = args[0]
            super().__init__(message)
            self.message = message
