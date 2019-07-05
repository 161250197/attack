from util.attack_util import create_attack_images
from util.fashion_mnist_util import create_test_data


def aiTest(images, shape):
    """
    后台测试只调用 aiTest 方法
    :param images: 一批照片 类型为 numpy.ndarray
    :param shape: 这批图片的 shape 类型为 tuple
    :return: Generate_images: 同输入图片相同的 shape 的修改后的批量图片数据
    """
    return create_attack_images(images)


def test_aiTest():
    """
    测试
    """
    (test_data, shape) = create_test_data()
    aiTest(test_data, shape)

test_aiTest()
