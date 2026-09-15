from skimage import data, io
import os

os.makedirs("dataset", exist_ok=True)

images = {
    "01_camera": data.camera(),
    "02_coins": data.coins(),
    "03_page": data.page(),
    "04_text": data.text(),
    "05_brick": data.brick(),
    "06_moon": data.moon(),
    "07_clock": data.clock(),
    "08_rocket": data.rocket(),
    "09_coffee": data.coffee()
}

for name, img in images.items():
    io.imsave(f"dataset/{name}.png", img)

print("Đã tạo dataset thành công!")