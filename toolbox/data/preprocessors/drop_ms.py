class DropMs():
    def run(self, data):
        data.index = data.index.map(lambda i: i.replace(microsecond=0))
        return data