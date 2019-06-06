# captcha_yh
基于CNN，keras的验证码识别项目

## 一.样本制造
首先我们要先制造样本，这里我们制造的样本去除了大小写的o,i和l，加上0-9一共是56个字符。训练的时候大约制造了3万张样本用做训练，2000张用做验证

```
import cv2
import numpy as np

line_num = 5
#图片个数
pic_num = 200
path = "validate/"


def randcolor():
    return (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    # return (0, 0, 0)


def randchar():
    return np.random.choice([chr(np.random.randint(65, 91)), chr(np.random.randint(97, 123)) ,np.random.randint(0, 9)])


def randpos(x_start, x_end, y_start, y_end):
    return (np.random.randint(x_start, x_end),
            np.random.randint(y_start, y_end))


img_heigth = 60
img_width = 240
for i in range(pic_num):
    img_name = ""
    # 生成一个随机矩阵，randint(low[, high, size, dtype])
    img = np.random.randint(np.random.randint(50, 100), np.random.randint(100, 150), (img_heigth, img_width, 3), np.uint8)
    # 显示图像
    cv2.imshow("ranImg",img)
    x_pos = 0
    y_pos = 25
    for i in range(4):
        char = randchar()
        while char == 'i' or char == 'I' or char == 'l' or char == 'L' or char == 'o' or char == 'O':
            char = randchar()
        img_name += char
        #各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        cv2.putText(img, char,
                    (np.random.randint(x_pos, x_pos + 20), np.random.randint(y_pos, y_pos + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    randcolor(),
                    3,
                    cv2.LINE_AA)
        x_pos += 45

    cv2.imshow("res",img)

    # 添加线段
    for i in range(line_num):
        img = cv2.line(img,
                       randpos(0, img_width, 0, img_heigth),
                       randpos(0, img_width, 0, img_heigth),
                       randcolor(),
                       np.random.randint(1, 2))

    cv2.imshow("line",img)
    cv2.imwrite(path + img_name + ".jpg", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
```
制造的样本如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019060615243145.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190606152448751.png)
## 二. 构建CNN神经网络并开始训练
```
import numpy as np
import os

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, concatenate ,BatchNormalization
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import glob

# 验证码所包含的字符 _表示未知
captcha_word = ['0','1','2','3','4','5','6','7','8','9',
           'A','B','C','D','E','F','G','H','J',
           'K','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
           'a','b','c','d','e','f','g','h','j','k','m','n','p',
           'q','r','s','t','u','v','w','x','y','z'
           ]

# 图片的长度和宽度
width = 240
height = 60

# 每个验证码所包含的字符数
word_len = 4
# 字符总数
word_class = len(captcha_word)

samples = glob.glob(r'train/*.jpg')

# 验证码素材目录
train_dir = 'data/train'

# 生成字符索引，同时反向操作一次，方面还原
char_indices = dict((c, i) for i, c in enumerate(captcha_word))
indices_char = dict((i, c) for i, c in enumerate(captcha_word))


# 验证码字符串转数组
def captcha_to_vec(captcha):
    # 创建一个长度为 字符个数 * 字符种数 长度的数组
    vector = np.zeros(word_len * word_class)

    # 文字转成成数组
    for i, ch in enumerate(captcha):
        idex = i * word_class + char_indices[ch]
        vector[idex] = 1
    return vector


# 把数组转换回文字
def vec_to_captcha(vec):
    text = []
    # 把概率小于0.5的改为0，标记为错误
    vec[vec < 0.5] = 0

    char_pos = vec.nonzero()[0]

    for i, ch in enumerate(char_pos):
        text.append(captcha_word[ch % word_class])
    return ''.join(text)

# 自定义评估函数
def custom_accuracy(y_true, y_pred):
    predict = tf.reshape(y_pred, [-1, word_len, word_class])
    max_idx_p = tf.argmax(predict, 2)#这个做法牛逼，不用再做stack和reshape了，2，是在Charset那个维度上
    max_idx_l = tf.argmax(tf.reshape(y_true, [-1, word_len,word_class]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    _result = tf.map_fn(fn=lambda e: tf.reduce_all(e),elems=correct_pred,dtype=tf.bool)
    return tf.reduce_mean(tf.cast(_result, tf.float32))

#获取目录下样本列表
image_list = []

#
for item in os.listdir(train_dir):
    image_list.append(item)
np.random.shuffle(image_list)
#创建数组，储存图片信息。样本个数、宽度和高度。
# 3代表图片的通道数，如果对图片进行了灰度处理，可以改为单通道 1
X = np.zeros((len(image_list), height, width, 3), dtype = np.uint8)
# 创建数组，储存标签信息
y = np.zeros((len(image_list), word_len * word_class), dtype = np.uint8)

for i,img in enumerate(image_list):
    if i % 10000 == 0:
        print(i)
    img_path = train_dir + "/" + img
    #读取图片
    raw_img = image.load_img(img_path, target_size=(height, width))
    #讲图片转为np数组
    X[i] = image.img_to_array(raw_img)
    #讲标签转换为数组进行保存
    y[i] = captcha_to_vec(img.split('.')[0])

#创建输入，结构为 高，宽，通道
input_tensor = Input( shape=(height, width, 3))

x = input_tensor

#构建卷积网络
#两层卷积层，一层池化层，重复3次。因为生成的验证码比较小，padding使用same
x = Convolution2D(32, 3, padding='same', activation='relu')(x)
x = Convolution2D(32, 3, padding='same', activation='relu')(x)
# x= BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)


x = Convolution2D(64, 3, padding='same', activation='relu')(x)
x = Convolution2D(64, 3, padding='same', activation='relu')(x)
# x= BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Convolution2D(128, 3, padding='same', activation='relu')(x)
x = Convolution2D(128, 3, padding='same',activation='relu')(x)
# x= BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

#Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
x = Flatten()(x)
#为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时随机断开一定百分比（rate）的输入神经元，Dropout层用于防止过拟合。
x = Dropout(0.25)(x)
x= BatchNormalization()(x)
#Dense就是常用的全连接层
#最后连接4个分类器，每个分类器是56个神经元，分别输出56个字符的概率。
x = [Dense(word_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(word_len)]
# x = [Dense(word_class, activation='sigmoid', name='c%d'%(i+1))(x) for i in range(word_len)]
output = concatenate(x)
#构建模型
model = Model(inputs=input_tensor, outputs=output)
# model.add(BatchNormalization())
#因为训练可能需要数个小时，所以这里加载了之前我训练好的参数。准确率为94%
#可以直接使用此参数继续进行训练，也可以自己从头开始训练
# model.load_weights('model/weights.10--6.55-0.3062.hdf5')

#这里优化器选用Adadelta，学习率0.1
opt = Adadelta(lr=0.1)
#编译模型以供训练，损失函数使用 categorical_crossentropy，使用accuracy评估模型在训练和测试时的性能的指标
#这里使用自定义的评估模型
model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics=['accuracy',custom_accuracy])
#每次epoch都保存一下权重，用于继续训练
checkpointer = ModelCheckpoint(filepath="model/weights.{epoch:02d}--{val_loss:.2f}-{val_acc:.4f}.hdf5",
                               verbose=2, save_weights_only=True)
#开始训练，validation_split代表10%的数据不参与训练，用于做验证急
#我之前训练了50个epochs以上，这里根据自己的情况进行选择。如果输出的val_acc已经达到你满意的数值，可以终止训练
model.fit(X, y, epochs= 10,callbacks=[checkpointer], validation_split=0.1)
# plot model
plot_model(model, to_file='model/model.png', show_shapes=True)

#保存权重和模型
model.save_weights('model/captcha_model_weights.h5')
model.save('model/captcha__model_2.h5')
```
得到的模型图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190606152525483.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNTE5NDE1,size_16,color_FFFFFF,t_70)
部分损失函数：
Loss：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190606153026253.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNTE5NDE1,size_16,color_FFFFFF,t_70)
custom_accuracy
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190606153106972.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNTE5NDE1,size_16,color_FFFFFF,t_70)
## 3. 结果验证
```
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
   ```

得到的结果：  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190606152610100.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNTE5NDE1,size_16,color_FFFFFF,t_70)
