import os
import random

path = r'./17flowers/jpg/'


def generate_txt(path):
    dir_list = os.listdir(path)
    print("======start=====")
    trainList = open(path + '/' + 'train.txt', 'w')
    valList = open(path + '/' + 'test.txt', 'w')
    for i in range(len(dir_list)):
        label_path = os.path.join(path, dir_list[i])
        if os.path.isdir(label_path):
            label = dir_list[i]
            files = os.listdir(label_path)
            random.shuffle(files)
            for num, file in enumerate(files):
                fileType = os.path.split(file)
                if fileType[1] == '.txt':
                    continue
                name = label_path + '/' + file + ' ' + label + '\n'
                if num <= int(len(files) * 0.7):
                    trainList.write(name)
                else:
                    valList.write(name)
    trainList.close()
    valList.close()
    print("======endall=======")


if __name__ == '__main__':
    generate_txt(path)





