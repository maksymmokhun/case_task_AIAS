"""Завантаження сирих даних зі знімка, на якому отримано всі числа в reports/ і README.

Кожен файл береться з фіксованого коміту (не з гілки main, яка оновлюється щодня) і перевіряється за git-хешем блоба,
тож чужі й пізніші версії не пройдуть. Лише стандартна бібліотека.
Запуск: python src/download_data.py   (з кореня проєкту; наявні файли з правильним хешем пропускаються)
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data/raw"

# (ім'я файлу, репозиторій, коміт, шлях у репозиторії, розмір у байтах, git blob sha1)
FILES = [
    ("volunteer_data_en.csv", "Vadimkin/ukrainian-air-raid-sirens-dataset", "7ccd4367b7057a4541ce109ca09cf1f0220f6a75",
     "datasets/volunteer_data_en.csv", 8892014, "6d37021ca5a39083466f8e43da504419fa87c9b7"),
    ("official_data_en.csv", "Vadimkin/ukrainian-air-raid-sirens-dataset", "58ee75bc20113181b9ddf8029d4cbf3a62d8cd10",
     "datasets/official_data_en.csv", 31025633, "4c9ef9bf023fefcbfd50cf45188ef36b5119aebe"),
    ("alerts_in_ua.csv", "JustJimer/ukraine-air-alert-stats", "3b0e7f1dc7e0e04e85db73f8a24b3e587d3e05ee",
     "archive/alerts_in_ua.csv", 2903971, "16a1d0a5efa79a3022777124b27601ebf75064b1"),
]


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, repo, commit, path, size, sha in FILES:
        dst = RAW / name
        if dst.exists() and git_blob_sha1(dst.read_bytes()) == sha:
            print(f"{name}: вже є, хеш збігається")
            continue
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        print(f"{name}: завантаження {url} ({size / 1e6:.1f} МБ)")
        data = urllib.request.urlopen(url, timeout=120).read()
        if len(data) != size or git_blob_sha1(data) != sha:
            sys.exit(f"{name}: розмір або хеш не збігаються з очікуваними, файл не збережено")
        dst.write_bytes(data)
        print(f"{name}: збережено, хеш збігається")


if __name__ == "__main__":
    main()
