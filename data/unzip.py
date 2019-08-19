import sys
if sys.version_info >= (3, 6):
    import zipfile
else:
    import zipfile36 as zipfile
import os

a = os.listdir(os.getcwd())


for i in a:
    with zipfile.ZipFile('/home/j/jifang/IEEE-CIS-Fraud-Detection/data/train_transaction.csv.zip', 'r') as zip_ref:
        zip_ref.extractall()