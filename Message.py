class Message:
    def __init__(self, type, message):
        self.content = message
        self.type = type

    def __str__(self):
        return self.content