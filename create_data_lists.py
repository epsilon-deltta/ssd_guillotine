from utils import create_data_lists
import os
if __name__ == '__main__':
    voc07 = os.path.abspath('./VOCdevkit/VOC2007')
    voc12 = os.path.abspath('./VOCdevkit/VOC2012')
    create_data_lists(voc07_path=voc07,
                      voc12_path=voc12,
                      output_folder='./')
