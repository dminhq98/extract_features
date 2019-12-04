import  os
FJoin = os.path.join
def get_files(path):
    file_list, dir_list = [], []
    for dir, subdirs, files in os.walk(path):
        file_list.extend([FJoin(dir, f) for f in files])
        dir_list.extend([FJoin(dir, d) for d in subdirs])
    file_list = filter(lambda x: not os.path.islink(x), file_list)
    dir_list = filter(lambda x: not os.path.islink(x), dir_list)
    return file_list, dir_list


if __name__ == "__main__":

    # main()
    path_image = "/home/dminhq98/PycharmProjects/extract_features/dataset"
    images = get_files(path_image)
    images=images[0]
    print(images)