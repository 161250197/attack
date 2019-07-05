import copy
import time
import numpy as np

from util.local_util import load_model, load_shadow_arrays
from util.model_util import show_images, predict_image_label, image_label_prob
from util.ssim import cal_ssim

__model = load_model()
__shadows = load_shadow_arrays()

__certain_prob = 0.6
__certain_ssim = 0.8
__secondary_ssim = 0.6
__create_time = 1

__show_attack_detail = True
__show_origin_attack_images = False


def create_attack_images(images):
    """
    创建对抗样本
    :param images: 一批照片 类型为 numpy.ndarray
    :return: Generate_images: 同输入图片相同的 shape 的修改后的批量图片数据
    """
    imgs = copy.deepcopy(images)

    start_time = time.time()

    img_count = len(imgs)
    for index in np.arange(img_count):
        print('[INFO] creating attack... [', index + 1, '/', img_count, ']')
        img = images[index]
        (label, probs) = predict_image_label(__model, img)

        max_ssim = -1

        target_shadows = [
            [1,3,4],
            [0,3,4],
            [0,3,4],
            [1,2,4],
            [0,2,3],
            [7,8,9],
            [0,2,3],
            [8,9],
            [2,4,9],
            [2,8]
        ]

        for shadow_label in target_shadows[label]:
            if shadow_label == 5 or shadow_label == label:
                continue
            (new_attack_img, new_ssim) = __create_attack_image(img, shadow_label)
            if new_ssim > max_ssim:
                imgs[index] = new_attack_img
                max_ssim = new_ssim

    # 展示效果
    __show_attack_effect(images, imgs, img_count)

    print('[INFO] use time(s) ', time.time() - start_time)

    return imgs


def __create_attack_image(image, shadow_label):
    """
    创建对抗样本
    :param image: 图片
    :param shadow_label: 遮罩标签
    :return: 对抗样本，ssim值
    """
    shadow = copy.deepcopy(__shadows[shadow_label])
    img = copy.deepcopy(image)

    shadow_percent = 0.5
    img_percent = 0.5

    new_img = combine_img(img * img_percent, shadow * shadow_percent)
    prob = image_label_prob(__model, new_img, shadow_label)

    start = time.time() + 1

    check_ssim = True
    check_secondary_ssim = False

    while True:
        if prob > __certain_prob:
            if not check_ssim:
                break
            else:
                target =  __certain_ssim
                if check_secondary_ssim:
                    target = __secondary_ssim

                ssim = cal_ssim(image, new_img)
                if ssim > target:
                    break
                else:
                    shadow_percent *= 0.8
                    img_percent *= 1.1
                    if img_percent > 1:
                        img_percent = 1
        else:
            shadow_percent *= 1.2
            if shadow_percent > 1:
                shadow_percent = 1
            img_percent *= 0.9

        if check_ssim and time.time() - start > __create_time:
            if check_secondary_ssim:
                check_ssim = False
            else:
                check_secondary_ssim = True
                start = time.time() + 1

        new_img = combine_img(img * img_percent, shadow * shadow_percent)
        prob = image_label_prob(__model, new_img, shadow_label)

    return new_img, cal_ssim(image, new_img)


def __show_attack_effect(origin, attack, img_count):
    """
    展示对抗样本效果
    :param origin: 原样本
    :param attack: 对抗样本
    :param img_count: 样本数
    :return:
    """
    suc_count = 0
    ssim_total = 0
    for i in np.arange(len(origin)):
        ssim = cal_ssim(origin[i], attack[i])
        ssim_total += ssim

        (ori_pred, ori_probs) = predict_image_label(__model, origin[i])
        (att_pred, att_probs) = predict_image_label(__model, attack[i])
        if ori_pred != att_pred:
            suc_count += 1

        if __show_attack_detail:
            print('----- \n', i, ' [ssim] ', ssim)
            print('[prob]')
            print(ori_pred, ' ', ori_probs[ori_pred], ' ', ori_probs[att_pred])
            print(att_pred, ' ', att_probs[ori_pred], ' ', att_probs[att_pred])

    print('[INFO] Result: success ', suc_count , '/', img_count)
    print('[INFO] average ssim ', ssim_total / img_count)

    if __show_attack_detail:
        shows = np.zeros(tuple((len(attack) * 2, 28, 28, 1)), dtype=float)
        for i in np.arange(len(attack)):
            shows[i * 2] += attack[i]
            shows[i * 2 + 1] += origin[i]
        show_images(shows)


def combine_img(img1, img2):
    img = img1 + img2
    for i in np.arange(len(img)):
        for j in np.arange(len(img[i])):
            if img[i][j] > 1:
                img[i][j] = 1
    return img
