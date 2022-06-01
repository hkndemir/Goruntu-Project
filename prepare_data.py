
import os,sys
from PIL import Image

cwd = os.getcwd()
train_path = os.path.join(cwd,'new_train_data')
dest_path = os.path.join(cwd,'raw_train_data')
new_dim = [32,32]


def create_image_data():
    if not os.path.exists(train_path):
            os.makedirs(train_path)
    if not os.path.exists(dest_path):
            os.makedirs(dest_path)

    class_list = []

    class_list.extend(os.listdir(train_path))

    for folder,val in enumerate(class_list):
        class_path = os.path.join(train_path,val)
        new_class_path = os.path.join(dest_path,val)

        if not os.path.exists(new_class_path):
                os.makedirs(new_class_path)

        image_list = os.listdir(class_path)

        for i in image_list:
            img_path = os.path.join(class_path,i)
            new = os.path.join(new_class_path,i)
            if not os.path.exists(new):
                print('Re-saving... '+str(i))
                image = Image.open(img_path)
                image = image.resize(new_dim)
                image = image.save(new)
            else:
                print('Found... '+str(i))
            

if __name__ == '__main__' :
	create_image_data()
	print('\nRe-saving Complete....')
