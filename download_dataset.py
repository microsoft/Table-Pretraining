import os
import requests
import tarfile
import zipfile
import shutil
"""
Download datasets used in our paper and move them into `raw_dataset` folder.
Now the script supports downloading the following datasets for different usage:

=== Fine-tuning ===
1. WikiSQL 
2. WikiTableQuestions
3. SQA
4. TabFact

=== Pre-training ===
1. Squall
"""
RAW_DATASET_FOLDER = "raw_dataset"


def download_file(url):
    """
    Download file into local file system from url
    """
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_filename


def download_wikisql():
    """
    Download WikiSQL dataset and unzip the files
    """
    WIKISQL_URL = "https://raw.github.com/salesforce/WikiSQL/master/data.tar.bz2"
    wikisql_raw_path = os.path.join(RAW_DATASET_FOLDER, "wikisql")
    wikisql_tar_file = download_file(WIKISQL_URL)
    # unzip and move it into raw_dataset folder
    tar = tarfile.open(wikisql_tar_file, "r:bz2")
    tar.extractall(wikisql_raw_path)
    tar.close()
    # remove the original file
    os.remove(wikisql_tar_file)


def download_wikitablequestion():
    """
    Download WikiSQL dataset and unzip the files
    """
    WTQ_URL = "https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip"
    wtq_raw_path = os.path.join(RAW_DATASET_FOLDER, "wtq")
    wtq_zip_file = download_file(WTQ_URL)
    # unzip and move it into raw_dataset folder
    with zipfile.ZipFile(wtq_zip_file) as zf:
        zf.extractall(RAW_DATASET_FOLDER)
    unzip_wtq_path = os.path.join(RAW_DATASET_FOLDER, "WikiTableQuestions")
    shutil.move(unzip_wtq_path, wtq_raw_path)
    # remove the original file
    os.remove(wtq_zip_file)


def download_sqa():
    """
    Download WikiSQL dataset and unzip the files
    """
    SQA_URL = "https://download.microsoft.com/download/1/D/C/1DC270D2-1B53-4A61-A2E3-88AB3E4E6E1F/SQA%20Release%201.0.zip"
    sqa_raw_path = os.path.join(RAW_DATASET_FOLDER, "sqa")
    sqa_zip_file = download_file(SQA_URL)
    # unzip and move it into raw_dataset folder
    with zipfile.ZipFile(sqa_zip_file) as zf:
        zf.extractall(RAW_DATASET_FOLDER)
    unzip_wtq_path = os.path.join(RAW_DATASET_FOLDER, "SQA Release 1.0")
    shutil.move(unzip_wtq_path, sqa_raw_path)
    # remove the original file
    os.remove(sqa_zip_file)


def download_tabfact():
    """
    Download WikiSQL dataset and unzip the files
    """
    SQA_URL = "https://download.microsoft.com/download/1/D/C/1DC270D2-1B53-4A61-A2E3-88AB3E4E6E1F/SQA%20Release%201.0.zip"
    sqa_raw_path = os.path.join(RAW_DATASET_FOLDER, "sqa")
    sqa_zip_file = download_file(SQA_URL)
    # unzip and move it into raw_dataset folder
    with zipfile.ZipFile(sqa_zip_file) as zf:
        zf.extractall(RAW_DATASET_FOLDER)
    unzip_wtq_path = os.path.join(RAW_DATASET_FOLDER, "SQA Release 1.0")
    shutil.move(unzip_wtq_path, sqa_raw_path)
    # remove the original file
    os.remove(sqa_zip_file)


def download_squall():
    """
    Download WikiSQL dataset and unzip the files
    """
    SQAULL_URL = "https://github.com/tzshi/squall/archive/refs/heads/main.zip"
    squall_raw_path = os.path.join(RAW_DATASET_FOLDER, "squall")
    squall_zip_file = download_file(SQAULL_URL)
    # unzip and move it into raw_dataset folder
    with zipfile.ZipFile(squall_zip_file) as zf:
        zf.extractall(RAW_DATASET_FOLDER)
    unzip_wtq_path = os.path.join(RAW_DATASET_FOLDER, "squall-main")
    shutil.move(unzip_wtq_path, squall_raw_path)
    # remove the original file
    os.remove(squall_zip_file)


if __name__ == '__main__':
    # download_wikisql()
    # download_wikitablequestion()
    download_sqa()
    # download_squall()
