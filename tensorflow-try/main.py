from util.attack_util import create_attack_images
from util.model_util import create_test_data


def aiTest(images, shape):
    """
    后台测试只调用 aiTest 方法
    :param images: 一批照片 类型为 numpy.ndarray
    :param shape: 这批图片的 shape 类型为 tuple
    :return: Generate_images: 同输入图片相同的 shape 的修改后的批量图片数据
    """
    imgs = images / 255
    attack_images = create_attack_images(imgs)
    result = attack_images * 255
    return result


def test_aiTest():
    """
    测试
    """
    (test_data, shape) = create_test_data()
    aiTest(test_data * 255, shape)

test_aiTest()
