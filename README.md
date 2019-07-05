# 深度学习测试

161250197 张李承

## 说明

* attack

  * main

    入口文件，有 aiTest 接口

    测试使用的 test_aiTest 方法

  * model.h5

    自己训练的模型

  * shadow.txt

    遮罩本地存储

  * util 工具包

    * attack_util 对抗模型生成脚本
    * local_util 本地存储相关脚本
    * model_util 模型训练相关脚本
    * ssim ssim计算脚本
    * fashion_model 模型训练包
      * fashion_mnist 模型训练脚本
      * pyimagesearch 模型包
        * minivggnet 模型脚本

## 思路

构建了自己的模型对所有的测试集做了分类，然后求平均算出中心点的图型遮罩，对传入的图片使用不同的遮罩进行覆盖，同时求尽可能高的ssim

## 其他

* 模型和遮罩文件已经存储到本地，使用 fasion_mnist.py 脚本中的 init_model test_model create_shadow test_shadow 可以生成和测试模型和遮罩
  * 因为 4 5 6 分类的模型分类不准确，所有遮罩生成时未使用所有的测试集
  * 遮罩和模型的测试有使用 cv2 进行可视化展示，但是需要的 imutils 包说明文件未明确说明可用，已被注释，如果需要查看可视化效果，请先去 model_util.py 文件中处理 import 和 TODO 部分的注释代码
* 对抗脚本生成时有输出提示信息， attack_util 中使用私有变量进行输出控制
  * __show_loading 输出进度情况
  * __show_attack_detail 统一输出 ssim 值 和 概率情况
  * __show_origin_attack_images 完成后可视化展示原图片和对抗图片
     **（数据量大时请不要开启）**