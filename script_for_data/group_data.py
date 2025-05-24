from __future__ import annotations
import os
import re
import sys
import shutil


def strip_view_and_extension(file_name: str) -> str:
    return re.sub(r'_(front|top|side)\.svg$|\.txt$|\.svg$', '', file_name)


def group_data_by_name(src_dir: str, dest_dir: str) -> None:
    processed: set[str] = set()
    for file in os.listdir(src_dir):
        obj_name: str = strip_view_and_extension(file)
        if obj_name not in processed:
            new_folder_dir: str = os.path.join(dest_dir, obj_name)
            os.mkdir(new_folder_dir)
            processed.add(obj_name)
        shutil.copy(
            src=os.path.join(src_dir, file),
            dst=os.path.join(dest_dir, obj_name),
        )


def main(argv: list[str]) -> None:
    if len(argv) < 3:
        print('Error: too few command-line arguments.')
    src_dir = argv[1]
    dest_dir = argv[2]
    group_data_by_name(src_dir, dest_dir)


if __name__ == '__main__':
    main(sys.argv)
