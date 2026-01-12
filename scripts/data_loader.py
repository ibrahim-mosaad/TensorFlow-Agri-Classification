import os
import tarfile
import httpx
from tensorflow.keras.preprocessing.image import ImageDataGenerator

async def download_and_prepare_data(url, extract_path="./data"):
    os.makedirs(extract_path, exist_ok=True)
    tar_path = os.path.join(extract_path, "dataset.tar")
    
    if not os.path.exists(tar_path):
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True)
            with open(tar_path, "wb") as f:
                f.write(response.content)
    
    with tarfile.open(tar_path, 'r:*') as tar:
        tar.extractall(path=extract_path)
    return os.path.join(extract_path, "images_dataSAT")

def get_data_generators(dataset_path, img_size=(64, 64), batch_size=64):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2
    )
    
    train_gen = datagen.flow_from_directory(
        dataset_path, target_size=img_size, batch_size=batch_size,
        class_mode='binary', subset='training'
    )
    
    val_gen = datagen.flow_from_directory(
        dataset_path, target_size=img_size, batch_size=batch_size,
        class_mode='binary', subset='validation', shuffle=False
    )
    return train_gen, val_gen