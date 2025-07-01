## This is a class for the testing motor setup (DummyMotor)
## Usage: run_first_image, run_all_images, center_pin

class DummyMotor():
    def __init__(self, name):
        self.name = name

    def move(self, *args, **kwargs):
        print(f"DummyMotor.move called on {self.name}")