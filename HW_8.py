import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split


original_dataset_dir = 'dataset_original'
base_dir = 'dataset'


if not os.path.exists(base_dir):
    os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

categories = ['cats', 'dogs', 'birds']

for category in categories:
   
    os.mkdir(os.path.join(train_dir, category))
    os.mkdir(os.path.join(validation_dir, category))
    os.mkdir(os.path.join(test_dir, category))

    
    category_dir = os.path.join(original_dataset_dir, category)
    filenames = [f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f))]

   
    train_files, test_files = train_test_split(filenames, test_size=0.4, random_state=42)
    validation_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

    
    def process_and_copy_files(file_list, destination_dir):
        for filename in file_list:
            file_path = os.path.join(category_dir, filename)
            img = Image.open(file_path)
            img = img.resize((224, 224))  
            img.save(os.path.join(destination_dir, filename))

    
    process_and_copy_files(train_files, os.path.join(train_dir, category))
    process_and_copy_files(validation_files, os.path.join(validation_dir, category))
    process_and_copy_files(test_files, os.path.join(test_dir, category))

print("Dataset successfully split into training, validation, and test sets.")
