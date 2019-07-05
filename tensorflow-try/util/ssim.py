_width = 28
_height = 28

def average(img):
    _avg = 0.0
    for i in range(_width):
        for j in range(_height):
            _avg += img[i][j]
    return _avg / (_width * _height)


def deviation(img):
    _avg = average(img)
    _deviation = 0.0
    for i in range(_width):
        for j in range(_height):
            _dev = img[i][j] - _avg
            _deviation += _dev ** 2
    return (_deviation / (_width * _height - 1)) ** 0.5


def cal_deviation(img1, img2):
    _avg1 = average(img1)
    _avg2 = average(img2)
    _deviation = 0.0
    for i in range(_width):
        for j in range(_height):
            _deviation += (img1[i][j] - _avg1) * (img2[i][j] - _avg2)
    return _deviation / (_width * _height - 1)


def SSIM(img1, img2):
    K1 = 0.01
    K2 = 0.03
    L = 255
    C1 = (K1 / L) * (K1 / L)
    C2 = (K2 / L) * (K2 / L)
    C3 = C2 / 2
    _average1 = average(img1)
    _average2 = average(img2)

    L = (2 * _average1 * _average2 + C1) / (_average1 ** 2 + _average2 ** 2 + C1)

    _deviation1 = deviation(img1)
    _deviation2 = deviation(img2)

    C = (2 * _deviation1 * _deviation2 + C2) / (_deviation1 * _deviation1 + _deviation2 * _deviation2 + C2)

    _deviation12 = cal_deviation(img1, img2)

    S = (_deviation12 + C3) / (_deviation1 * _deviation2 + C3)

    # print(L,":",C,":",S,":",L*C*S)
    return L * C * S
