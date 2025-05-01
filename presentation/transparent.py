from PIL import Image as PILImage
import numpy as np


def make_transparent(image_path):
    image = PILImage.open(image_path)
    image = np.array(image.convert("RGBA"))
    # White to transparent
    white_areas = np.where(image[:, :, :3] >= [245, 245, 245])
    image[white_areas[0], white_areas[1], 3] = 0
    image = PILImage.fromarray(image)
    image.save(image_path.replace(".png", "_transparent.png"))


if __name__ == "__main__":
    make_transparent("./bulldozer.png")

