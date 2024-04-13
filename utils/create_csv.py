# Script to create CSV data file from Pascal VOC annotation files
# Based off code from GitHub user datitran: https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py

import os
import sys
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    try:
        xml_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            for member in root.findall('object'):
                class_name = member[0].text
                xmin = int(member.find('bndbox/xmin').text)
                ymin = int(member.find('bndbox/ymin').text)
                xmax = int(member.find('bndbox/xmax').text)
                ymax = int(member.find('bndbox/ymax').text)
                xml_list.append((filename, width, height, class_name, xmin, ymin, xmax, ymax))
        
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        return xml_df
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def main():

    if len(sys.argv) != 2:
        print("Usage: python script.py <path>")
        return

    # Get the path argument
    pato = sys.argv[1]

    if os.path.exists(pato) and os.path.isdir(pato):
        image_path = os.path.join(os.getcwd(), (pato))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv((pato + '_labels.csv'), index=None)
        print('Successfully converted xml to csv.')
    else:
        print("Folder does not exist.")

main()