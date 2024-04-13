import argparse
from pathlib import Path
import random
import os
import sys

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Process some strings.')

    # Add arguments
    parser.add_argument('--images', type=str, help='Assign input to train')
    parser.add_argument('--train', type=float, help='Assign input to train')
    parser.add_argument('--val', type=float, help='Assign input to validation')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Define paths to image folders
    image_path = args.images

     # Find the index of the last occurrence of the delimiter
    last_index = image_path.rfind("\\")

    # If the delimiter is not found, return the original string
    if last_index == -1:
        nuevacarpeta = image_path
    else:
        nuevacarpeta = image_path[:last_index]

    train_path = nuevacarpeta + "/train"
    val_path = nuevacarpeta + "/validation"
    test_path = nuevacarpeta + "/test"

    os.makedirs(train_path)
    os.makedirs(val_path)
    os.makedirs(test_path)

    # Get list of all images
    jpeg_file_list = [path for path in Path(image_path).rglob('*.jpeg')]
    jpg_file_list = [path for path in Path(image_path).rglob('*.jpg')]
    png_file_list = [path for path in Path(image_path).rglob('*.png')]
    bmp_file_list = [path for path in Path(image_path).rglob('*.bmp')]

    if sys.platform == 'linux':
        JPEG_file_list = [path for path in Path(image_path).rglob('*.JPEG')]
        JPG_file_list = [path for path in Path(image_path).rglob('*.JPG')]
        file_list = jpg_file_list + JPG_file_list + png_file_list + bmp_file_list + JPEG_file_list + jpeg_file_list
    else:
        file_list = jpg_file_list + png_file_list + bmp_file_list + jpeg_file_list

    file_num = len(file_list)
    print('Total images: %d' % file_num)

    # Determine number of files to move to each folder
    train_percent = args.train  # 80% of the files go to train
    val_percent = args.val # 10% go to validation
    
    train_num = int(file_num*train_percent)
    val_num = int(file_num*val_percent)
    test_num = file_num - train_num - val_num

    print('Images moving to train: %d' % train_num)
    print('Images moving to validation: %d' % val_num)
    print('Images moving to test: %d' % test_num)

    # Select 80% of files randomly and move them to train folder

    for i in range(train_num):
        move_me = random.choice(file_list)
        fn = move_me.name
        base_fn = move_me.stem
        parent_path = move_me.parent
        xml_fn = base_fn + '.xml'
        os.rename(move_me, train_path+'/'+fn)
        os.rename(os.path.join(parent_path,xml_fn),os.path.join(train_path,xml_fn))
        file_list.remove(move_me)

    # Select 10% of remaining files and move them to validation folder
    for i in range(val_num):
        move_me = random.choice(file_list)
        fn = move_me.name
        base_fn = move_me.stem
        parent_path = move_me.parent
        xml_fn = base_fn + '.xml'
        os.rename(move_me, val_path+'/'+fn)
        os.rename(os.path.join(parent_path,xml_fn),os.path.join(val_path,xml_fn))
        file_list.remove(move_me)

    # Move remaining files to test folder
    for i in range(test_num):
        move_me = random.choice(file_list)
        fn = move_me.name
        base_fn = move_me.stem
        parent_path = move_me.parent
        xml_fn = base_fn + '.xml'
        os.rename(move_me, test_path+'/'+fn)
        os.rename(os.path.join(parent_path,xml_fn),os.path.join(test_path,xml_fn))
        file_list.remove(move_me)

    

if __name__ == "__main__":
    main()
