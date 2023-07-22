import struct
from random import choice, shuffle
from modules.nn import DataPoint


class MNIST:
    def __init__(self):
        with open('mnist_data/train-images.idx3-ubyte', 'rb') as f:
            f.read(4)
            img_count: int = struct.unpack('>i', f.read(4))[0]
            img_size: int = struct.unpack('>i', f.read(4))[0] * struct.unpack('>i', f.read(4))[0]
            img_values: int = img_count * img_size
            img_data: list[float] = list(struct.unpack('>' + 'B' * img_values, f.read(img_values)))

        with open('mnist_data/train-labels.idx1-ubyte', 'rb') as f:
            f.read(4)
            lbl_count: int = struct.unpack('>i', f.read(4))[0]
            lbl_data: list[int] = list(struct.unpack('>' + 'B' * lbl_count, f.read(lbl_count)))

        self.images: list[DataPoint] = []

        for i in range(min(img_count, lbl_count)):
            data = img_data[i * img_size:(i + 1) * img_size]
            label = lbl_data[i]
            new_img = DataPoint(data, [float(i == label) for i in range(10)])
            self.images.append(new_img)

    def __getitem__(self, s: slice) -> list[DataPoint]:
        return self.images[s]

    def get_random_image(self) -> DataPoint:
        return choice(self.images)

    def get_random_images(self, count: int) -> list[DataPoint]:
        shuffle(self.images)
        return self.images[:count]
