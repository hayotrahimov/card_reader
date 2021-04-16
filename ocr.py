def try_this():
    # Импорт инструментария
    from imutils import contours
    import numpy as np
    import argparse
    import cv2
    import myutils

    # Установить параметры
    image = "images/image1.png"
    font = "images/shrift1.png"
    # image = "images/uzcard-milliy.png"
    # font = "images/font-proximanova.png"
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default=image, help="path to input image")
    ap.add_argument("-t", "--template", default=font, help="path to template OCR-A image")
    args = vars(ap.parse_args())

    # Укажите тип кредитной карты
    FIRST_NUMBER = {
        "3": "American Express",
        "4": "Visa",
        "5": "MasterCard",
        "6": "Discover Card"
    }

    # Отображение рисунка
    def cv_show(name, img):
        # return
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Прочитать шаблон изображения
    img = cv2.imread(args["template"])
    # cv_show('img', img)
    # Изображение в градациях серого
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv_show('ref', ref)
    # Двоичное изображение
    ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
    cv_show('ref', ref)

    # Рассчитать контур
    # Параметр, принятый
    # функцией  # cv2.findContours (), является двоичным изображением, то есть черно-белым (не в градациях серого), cv2.RETR_EXTERNAL определяет только внешний контур, а cv2.CHAIN_APPROX_SIMPLE сохраняет только координаты конечной точки
    # Каждый элемент в возвращаемом списке является контуром на изображении

    refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
    # cv_show('img', img)
    # print(np.array(refCnts).shape)
    refCnts = myutils.sort_contours(refCnts, method="слева направо")[0]  # сортировка слева направо, сверху вниз
    digits = {}

    # Обходить каждый контур
    for (i, c) in enumerate(refCnts):
        # Рассчитать описанный прямоугольник и изменить его размер до подходящего размера
        (x, y, w, h) = cv2.boundingRect(c)
        roi = ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        # Каждый номер соответствует каждому шаблону
        digits[i] = roi

    # Инициализировать ядро ​​свертки
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Чтение входного изображения, препроцесс
    image = cv2.imread(args["image"])
    # cv_show('image', image)
    image = myutils.resize(image, width=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv_show('gray', gray)

    # Верхняя операция, чтобы выделить более яркие области
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    # cv_show('tophat', tophat)
    #
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,  # ksize = -1 эквивалентно 3 * 3
                      ksize=-1)

    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")

    print(np.array(gradX).shape)
    # cv_show('gradX', gradX)

    # Соедините числа, закрыв операцию (сначала расширение, затем коррозия)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    # cv_show('gradX', gradX)
    # THRESH_OTSU автоматически найдет подходящий порог, подходящий для двойных пиков, вам нужно установить для параметра threshold 0
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv_show('thresh', thresh)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)  # Другая закрытая операция
    # cv_show('thresh', thresh)

    # Рассчитать контур

    threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)

    cnts = threshCnts
    cur_img = image.copy()
    cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
    cv_show('img', cur_img)
    locs = {}
    cards = []
    not_cards = []

    # Пройдите по контуру
    for (i, c) in enumerate(cnts):
        # Рассчитать прямоугольник
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # Выберите соответствующую область, в соответствии с фактической задачей, здесь в основном группа из четырех чисел
        if ar > 3.0 and ar < 4.0:
            if (w >= 30 and w < 45) and (h > 7 and h < 15):
                print(ar, x, y, w, h)
                # Познакомьтесь с пребыванием
                a = (x, y, w, h)
                if y not in locs.keys():
                    locs[y] = []
                locs[y].append(a)
    for locs_ in locs.keys():
        if len(locs[locs_]) >= 4:
            cards = locs[locs_]
    print(cards, not_cards)
    # Сортировка соответствующих контуров слева направо
    cards = sorted(cards, key=lambda x: x[0])
    output = []
    print(cards)

    # Пройдите по номерам в каждом наброске
    for (i, (gX, gY, gW, gH)) in enumerate(cards):
        # initialize the list of group digits
        groupOutput = []

        # Извлечь каждую группу в соответствии с координатами
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        # cv_show('group', group)
        # Предварительная обработка
        group = cv2.threshold(group, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv_show('group', group)
        # Рассчитать план каждой группы
        digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

        # Рассчитать каждое значение в каждой группе
        for c in digitCnts:
            # Найти контур текущего значения и изменить его размер до подходящего размера
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))
            # cv_show('roi', roi)

            # Рассчитать счет матча
            scores = []

            # Рассчитать каждый счет в шаблоне
            for (digit, digitROI) in digits.items():
                # Шаблон соответствия
                result = cv2.matchTemplate(roi, digitROI,
                                           cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)

            # Получить наиболее подходящий номер
            groupOutput.append(str(np.argmax(scores)))

        # Нарисуй это
        cv2.rectangle(image, (gX - 5, gY - 5),
                      (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
        cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # Получить результат
        output.extend(groupOutput)

    # Печать результата
    if output:
        print("Credit Card Type: {}".format(FIRST_NUMBER.get(output[0]) or "unknown"))
        print("Credit Card #: {}".format("".join(output)))
    cv_show("Image", image)
