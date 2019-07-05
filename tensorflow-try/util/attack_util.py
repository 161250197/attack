import copy
from util.model_util import load_model, load_selection, load_shadow_arrays
from util.fashion_mnist_util import show_images, predict_image_label, image_label_prob
from util.ssim import SSIM
import numpy as np
import time

_model = load_model()
_shadows = load_shadow_arrays()
_certain_prob = 0.8
_certain_ssim = 0.8
_create_time = 6
_show_attack_detail = True


def create_attack_images(images):
    """
    创建对抗样本
    :param images: 一批照片 类型为 numpy.ndarray
    :param shape: 这批图片的 shape 类型为 tuple
    :return: Generate_images: 同输入图片相同的 shape 的修改后的批量图片数据
    """
    imgs = copy.deepcopy(images)

    img_count = len(imgs)
    for i in np.arange(img_count):
        print('[INFO] creating attack... [', i + 1, '/', img_count, ']')
        img = images[i]
        (label, probs) = predict_image_label(_model, images[i])

        max_ssim = -1
        for shadow_label in np.arange(10):
            if shadow_label == 5 or shadow_label == label:
                continue
            (new_attack_img, new_ssim) = create_attack_image(img, shadow_label)
            if new_ssim > max_ssim:
                imgs[i] = new_attack_img
                max_ssim = new_ssim

    # 展示效果
    __show_attack_effect(images, imgs, img_count)

    return imgs


def __show_attack_effect(origin, attack, img_count):
    suc_count = 0
    ssim_total = 0
    for i in np.arange(len(origin)):
        ssim = SSIM(origin[i], attack[i])
        ssim_total += ssim

        (ori_pred, ori_probs) = predict_image_label(_model, origin[i])
        (att_pred, att_probs) = predict_image_label(_model, attack[i])
        if ori_pred != att_pred:
            suc_count += 1

        if _show_attack_detail:
            print('----- \n', i, ' [ssim] ', ssim)
            print('[prob]')
            print(ori_pred, ' ', ori_probs[ori_pred], ' ', ori_probs[att_pred])
            print(att_pred, ' ', att_probs[ori_pred], ' ', att_probs[att_pred])

    print('[INFO] Result: success ', suc_count , '/', img_count)
    print('[INFO] average ssim ', ssim_total / img_count)

    if _show_attack_detail:
        shows = np.zeros(tuple((len(attack) * 2, 28, 28, 1)), dtype=float)
        for i in np.arange(len(attack)):
            shows[i * 2] += attack[i]
            shows[i * 2 + 1] += origin[i]
        show_images(shows)


def create_attack_image(image, shadow_label):
    shadow = copy.deepcopy(_shadows[shadow_label])
    img = copy.deepcopy(image)

    shadow_percent = 1
    img_percent = 0

    new_img = combine_img(img * img_percent, shadow * shadow_percent)
    ssim = SSIM(image, new_img)
    prob = image_label_prob(_model, new_img, shadow_label)

    start = time.time()

    check_ssim = True

    while True:
        if prob > _certain_prob:
            if (not check_ssim) or ssim > _certain_ssim:
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

        if check_ssim and time.time() - start > _create_time:
            check_ssim = False

        new_img1 = combine_img(img * img_percent, shadow * shadow_percent)
        ssim1 = SSIM(image, new_img1)
        prob1 = image_label_prob(_model, new_img1, shadow_label)
        if ssim1 > ssim or prob1 > prob:
            ssim = ssim1
            prob = prob
            new_img = new_img1
        else:
            break

    return new_img, ssim


def combine_img(img1, img2):
    img = img1 + img2
    for i in np.arange(len(img)):
        for j in np.arange(len(img[i])):
            if img[i][j] > 1:
                img[i][j] = 1
    return img