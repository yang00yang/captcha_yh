import numpy as np
import os
import logging
from keras.models import load_model  # 一系列网络层按顺序构成的栈
import glob
from keras.preprocessing import image
import tensorflow as tf

logger = logging.getLogger("forecast by model")
# 每个验证码所包含的字符数
word_len = 4
image_path = 'validate/'
# image_path = '20190430/'
# 验证码所包含的字符 _表示未知
captcha_word = ['0','1','2','3','4','5','6','7','8','9',
           'A','B','C','D','E','F','G','H','J',
           'K','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
           'a','b','c','d','e','f','g','h','j','k','m','n','p',
           'q','r','s','t','u','v','w','x','y','z'
           ]

# 字符总数
word_class = len(captcha_word)

#日志初始化
def init_logger():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])

def custom_accuracy(y_true, y_pred):
    predict = tf.reshape(y_pred, [-1, word_len, word_class])
    max_idx_p = tf.argmax(predict, 2)#这个做法牛逼，不用再做stack和reshape了，2，是在Charset那个维度上
    max_idx_l = tf.argmax(tf.reshape(y_true, [-1, word_len,word_class]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    _result = tf.map_fn(fn=lambda e: tf.reduce_all(e),elems=correct_pred,dtype=tf.bool)
    return tf.reduce_mean(tf.cast(_result, tf.float32))

# load json and create model
def create_model():
    weight_path = 'model/captcha__model.h5'
    model = load_model(weight_path,
                        custom_objects={'custom_accuracy': custom_accuracy})
    return model

# 把数组转换回文字
def vec_to_captcha(vec):
    text = []
    # 把概率小于0.5的改为0，标记为错误
    vec[vec < 0.5] = 0

    char_pos = vec.nonzero()[0]

    for i, ch in enumerate(char_pos):
        text.append(captcha_word[ch % word_class])
    return ''.join(text)


if __name__ == '__main__':
    init_logger()
    model = create_model()
    image_list = []
    for item in os.listdir(image_path):
        image_list.append(item)
    np.random.shuffle(image_list)
    # 图片总数
    image_count = 0
    # 成功次数
    success_count = 0
    # 图片
    for i, img in enumerate(image_list):
        if i % 10000 == 0:
            print(i)
        img_path = image_path + img
        # 读取图片
        raw_img = image.load_img(img_path, target_size=(60, 240))
        code = img.replace('.jpg', '')
        code = code.split('_')[0]
        logger.debug('正确的验证码为' + code)
        X_test = np.zeros((1, 60, 240, 3), dtype=np.float32)
        X_test[0] = image.img_to_array(raw_img)
        # 预测
        predict = model.predict(X_test)
        n = 56  # 大列表中几个数据组成一个小列表
        arr = []
        arr.append(predict[0][0:56])
        arr.append(predict[0][56:112])
        arr.append(predict[0][112:168])
        arr.append(predict[0][168:224])
        predictions = []
        predictions.append(np.argmax(arr[0]))
        predictions.append(np.argmax(arr[1]))
        predictions.append(np.argmax(arr[2]))
        predictions.append(np.argmax(arr[3]))
        # predictions = np.argmax(predict, axis=1)
        # 标签字典
        keys = range(56)
        label_dict = dict(zip(keys, captcha_word))

        result = ''.join([label_dict[pred] for pred in predictions])
        image_count = image_count + 1
        if result == code:
            success_count  = success_count + 1
        logger.debug("预测的结果为" + result)
        logger.debug("目前正确率" + str(success_count / image_count))
    logger.debug("总次数" + str(image_count))
    logger.debug("成功次数" + str(success_count))
    logger.debug("正确率" + str(success_count/image_count))