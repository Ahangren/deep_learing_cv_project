import os,shutil
import random

root_dir=r'./data/PetImages'
categories=os.listdir(root_dir)

train_dir=os.path.join(root_dir,'train')
os.makedirs(train_dir)
test_dir=os.path.join(root_dir,'test')

test_rate=0.1

for category in categories:
    src_dir=os.path.join(root_dir,category)
    filenames=os.listdir(src_dir)
    test_num=int(len(filenames)*test_rate)
    test_filename=random.sample(filenames,test_num)

    test_category_dir=os.path.join(test_dir,category)
    os.makedirs(test_category_dir,exist_ok=True)
    for test_filename in test_filename:
        src_path=os.path.join(src_dir,test_filename)
        tgt_path=os.path.join(test_category_dir,test_filename)
        shutil.move(src_path,tgt_path)

    train_category_dir=os.path.join(train_dir,category)
    os.makedirs(train_category_dir,exist_ok=True)
    for train_filename in os.listdir(src_dir):
        src_path=os.path.join(src_dir,train_filename)
        tgt_path=os.path.join(train_category_dir,train_filename)
        shutil.move(src_path,tgt_path)

    os.rmdir(src_dir)
print("数据处理完成")
