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