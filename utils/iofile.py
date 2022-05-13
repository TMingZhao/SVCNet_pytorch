import os
from xml.etree import ElementTree
from lxml import etree
import numpy as np

from utils.pc_utils import save_ply

def create_dir(paths):
    if isinstance(paths, list):
        for path in paths:
            dir = os.path.split(path)[0]
            if not os.path.exists(dir):
                os.makedirs(dir)
    else:
        path = paths
        dir = os.path.split(path)[0]
        if not os.path.exists(dir):
            os.makedirs(dir) 

def create_dir_save_ply(xyz, path, cover=True):
    if os.path.exists(path) and not cover:
        return
    create_dir(path)
    save_ply(xyz, path)

# 对于任意层级的目录都可以索引
def get_file_list(file_name, type_='.ply'):
    filelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(type_):
                filelist.append(os.path.join(parent, filename))
    return filelist


# 读取XML文件，输出文件名、标签和位置
def read_xml(xml_path):
    parser = etree.XMLParser(encoding='utf-8')
    xmltree = ElementTree.parse(xml_path, parser=parser).getroot()

    box_list = []
    file_name = xmltree.find('filename').text
    for object_iter in xmltree.findall('object'):
        bndbox = object_iter.find("bndbox")
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
        label = object_iter.find('name').text
        box_list.append([file_name, label, xmin, ymin, xmax, ymax])


    return box_list

def read_ply_label(label_path):
    assert(os.path.exists(label_path)), "%s is not exist"%(label_path)
    name_label_list = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            content = line.rstrip().split(' ')
            img_name = content[0]
            bbox = [float(cont) for cont in content[1:]]
            name_label_list.append((img_name, bbox))
    
    return name_label_list

def dict_scale(dic, scale):
    for key, value in dic.items():
        dic[key] = dic[key] * scale
    return dic

def merge2dict(x, y):
    '''
    x, y:{key:value, ...}
    y += x
    '''
    if isinstance(x, list):
        for xi in x:
            for key,value in xi.items():
                if key in y.keys():
                    y[key] += value
                else:
                    y[key] = value
    else:
        for key,value in x.items():
            if key in y.keys():
                y[key] += value
            else:
                y[key] = value
    
            
def Linear_trainsform(arr, mm=255, mn=0):
    min_ = np.min(arr)
    max_ = np.max(arr)
    if min_ == max_:
        k = 1
    else:
        k = (mm - mn)/(max_ - min_)
    arr = (arr - min_) * k + mn
    return arr

def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

if __name__ == "__main__":
    list_ = read_ply_label("data/sketchfab/the-dark-knight/patch/00.txt")
    print(list_)