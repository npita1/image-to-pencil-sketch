import os

# Postavi putanju do foldera
folder_path = './dataset/original'

# Dobavi listu svih datoteka u folderu
files = os.listdir(folder_path)

# Sortiraj datoteke (osigurava redoslijed)
files = sorted(files)

# Iteriraj kroz datoteke i preimenuj ih
for i, filename in enumerate(files):
    # Dobavi ekstenziju datoteke
    file_ext = os.path.splitext(filename)[1]

    # Novi naziv datoteke (npr. 1.jpg, 2.jpg)
    new_name = f"{i + 1}{file_ext}"

    # Potpune putanje do stare i nove datoteke
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)

    # Preimenuj datoteku
    os.rename(old_path, new_path)

print("Sve slike su uspješno preimenovane!")
