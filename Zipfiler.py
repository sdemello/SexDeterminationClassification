import zipfile
with zipfile.ZipFile("TNI_Knees_RD_with_Gender.zip", "r") as zip_ref:
    zip_ref.extractall("./home/Projects/")



