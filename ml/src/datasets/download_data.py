import gdown
import os


def download_from_gdrive(file_id=None, file_url=None, output_path=None):
    """
    Download a file from Google Drive using either file ID or URL

    Args:
        file_id (str): The Google Drive file ID
        file_url (str): The complete Google Drive sharing URL
        output_path (str): Path where the file should be saved
    """
    try:
        if file_url:
            # Download using the complete URL
            gdown.download(file_url, output_path, quiet=False)
        elif file_id:
            # Download using file ID
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
        else:
            raise ValueError("Either file_id or file_url must be provided")

        if os.path.exists(output_path):
            print(f"Successfully downloaded to: {output_path}")
        else:
            print("Download failed")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Example usage:
if __name__ == "__main__":
    url = "https://drive.google.com/uc?export=download&id=1QgjwdQpY96NAVGdvjjFrXLhb48o15EO_"
    download_from_gdrive(file_url=url, output_path="./src/datasets/data.zip")
