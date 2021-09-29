import re
import uuid

import cv2
import easyocr as easyocr
import numpy as np
from aiohttp import web

cardnumber_pattern = re.compile("^([0-9]{4} *)*$")
expire_pattern = re.compile("^[0-9]{2}/[0-9]{2}$")
holder_pattern = re.compile("^[a-zA-Z '.-]*$")
reader = easyocr.Reader(['en'], gpu=False)
THRESHOLD = 0.6


def get_extracted_words(img, **kwargs):
    data = {"card_number": None, "holder": None, "expire": None}
    request_data: dict = dict(kwargs)
    for key in request_data.keys():
        if '_ths' in key:
            request_data[key] = float(request_data.get(key))
    holders = reader.readtext(img, allowlist="ABCDEFGHIKLMNOPQRSTVXYZabcdefghijklmnopqrstuvwxyz' ", **request_data)
    numbers = reader.readtext(img, allowlist="0123456789:/-", **request_data)
    for line in holders:
        print(line)
        if line[-1] > THRESHOLD:
            if holder_pattern.match(line[-2]):
                data['holder'] = line[-2]
    for line in numbers:
        print(line)
        if line[-1] > THRESHOLD:
            if cardnumber_pattern.match(line[-2]):
                if not data['card_number'] or len(data['card_number']) < len(line[-2]):
                    data['card_number'] = str(line[-2]).replace(" ", "")
            elif expire_pattern.match(line[-2]):
                data['expire'] = line[-2]
    print(data)
    return data


async def handle(request: web.Request):
    post = await request.post()
    image = post.get("image")
    data = dict(post)
    data.pop('image')
    img_str = image.file.read()
    nparr = np.frombuffer(img_str, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
    data = get_extracted_words(img_np, **data)
    return web.json_response(data)


if __name__ == '__main__':
    app = web.Application()
    app.add_routes([web.get('/', handle),
                    web.get('/{name}', handle),
                    web.post('/', handle),
                    ])
    web.run_app(app)
