class Traindata:
    label = None
    data = None

    def __init__(self, label, data):
        self.label = label
        self.data = data


class Testdata:
    data = None

    def __init__(self, data):
        self.data = data
