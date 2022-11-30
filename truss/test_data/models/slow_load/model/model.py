from time import sleep


class SlowLoad:
    def load(self):
        sleep(2)
        print("Finished loading slow model")
