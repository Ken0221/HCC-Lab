import gdown


if __name__ == "__main__":
    URL = "https://docs.google.com/uc?id={id}"
    FILES = [
        ('19zDhGAb7hDNKHV80qwwcdQvGcqVeWjyh', 'pokemon.zip'),
    ]
    for file_id, file_name in FILES:
        gdown.download(URL.format(id=file_id), file_name, quiet=False)
