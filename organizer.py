import os

PAGES_DIR = "pages"

def organize_pages(pages_dir):

    if not os.path.exists(pages_dir):
        print(f"Папка {pages_dir} не существует.")
        return

    for filename in os.listdir(pages_dir):
        file_path = os.path.join(pages_dir, filename)

        if not os.path.isfile(file_path):
            continue

        folder_name = os.path.splitext(filename)[0]
        new_folder_path = os.path.join(pages_dir, folder_name)

        os.makedirs(new_folder_path, exist_ok=True)

        new_file_path = os.path.join(new_folder_path, filename)
        os.rename(file_path, new_file_path)

        print(f"Файл {filename} перемещен в папку {folder_name}")

if __name__ == "__main__":
    organize_pages(PAGES_DIR)