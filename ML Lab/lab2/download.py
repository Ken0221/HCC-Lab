import gdown

if __name__ == "__main__":
    URL = "https://docs.google.com/uc?id={id}"
    FILES = [
        ('1TTEJLkcy0mIt23toLnCpoXsKh-522dc8', 'dataset.zip'),
        ('1sXvQbGulF-yZdzChyPGulUTg7rO7TZZE', 'pretrain.zip'),
    ]
    for file_id, file_name in FILES:
        gdown.download(URL.format(id=file_id), file_name, quiet=False)
