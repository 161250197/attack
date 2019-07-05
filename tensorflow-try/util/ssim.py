__width = 28
__height = 28


def __average(img):
    _avg = 0.0
    for i in range(__width):
        for j in range(__height):
            _avg += img[i][j]
    return _avg / (__width * __height)


def __deviation(img):
    _avg = __average(img)
    _deviation = 0.0
    for i in range(__width):
        for j in range(__height):
            _dev = img[i][j] - _avg
            _deviation += _dev ** 2
    return (_deviation / (__width * __height - 1)) ** 0.5


def __cal_deviation(img1, img2):
    _avg1 = __average(img1)
    _avg2 = __average(img2)
    _deviation = 0.0
    for i in range(__width):
        for j in range(__height):
            _deviation += (img1[i][j] - _avg1) * (img2[i][j] - _avg2)
    return _deviation / (__width * __height - 1)


def cal_ssim(img1, img2):
    """
    计算 ssim
    :param img1: 图片1
    :param img2: 图片2
    :return: ssim值
    """
    K1 = 0.01
    K2 = 0.03
    L = 255
    C1 = (K1 / L) * (K1 / L)
    C2 = (K2 / L) * (K2 / L)
    C3 = C2 / 2
    _average1 = __average(img1)
    _average2 = __average(img2)

    L = (2 * _average1 * _average2 + C1) / (_average1 ** 2 + _average2 ** 2 + C1)

    _deviation1 = __deviation(img1)
    _deviation2 = __deviation(img2)

    C = (2 * _deviation1 * _deviation2 + C2) / (_deviation1 * _deviation1 + _deviation2 * _deviation2 + C2)

    _deviation12 = __cal_deviation(img1, img2)

    S = (_deviation12 + C3) / (_deviation1 * _deviation2 + C3)

    return L * C * S
