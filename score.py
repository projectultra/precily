import requests

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

# Example usage
url = "https://drive.google.com/file/d/1CdT-hVtsNBt59K7QA5Vwso4u4ssHQi9l/view?usp=drive_link"  # Replace with the URL of the file you want to download
save_path = "embedding"   # Replace with the desired save path and filename

download_file(url, save_path)
