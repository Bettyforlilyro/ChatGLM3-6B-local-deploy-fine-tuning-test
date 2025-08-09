import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter


def find_txt_files(directory):
    txt_files = []

    # 遍历指定文件夹
    for root, dirs, files in os.walk(directory):
        # 遍历文件夹中的文件
        for file in files:
            # 检查文件是否以.txt结尾
            if file.endswith('.txt'):
                # 如果是.txt文件，将文件路径添加到列表中
                txt_files.append(os.path.join(root, file))

    return txt_files

def get_singel_jsonFile(file_path):
    import json

    context_list = []
    # 假设txt文件的内容是这样的，每一行都是一个json对象
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        for line in lines:
            # 将每一行的字符串转换为字典
            data = json.loads(line)

            # 提取并打印'inside'字段的值
            _line = (data.get('type') + "_" + data.get('inside'))
            context_list.append(_line)
    return context_list


import re


def merge_numbers(input_string):
    if input_string == "-100":
        return -100
    # 通过正则表达式匹配数字，包括整数和小数
    else:
        pattern = r"(\d+(?:,\d+)*(?:\.\d+)?)"

        try:
            match = re.search(pattern, input_string)
            if match:
                result = match.group(1)

            # 将最终结果转换为浮点数并返回
            result = result.replace(',', '')
            result = float(result)
            return result
        except:
            return -100

if __name__ == '__main__':

    file_path = "./alltxt/tsinghua.txt"
    dir_path = "./alltxt"
    loader = TextLoader(file_path=file_path, encoding='utf-8')
    pages = loader.load_and_split()
    # print(pages)
    chunk_size = 60     # 每段字数长度（文本分割的滑窗长度）
    chunk_overlap = 3   # 重叠的字数（重叠滑窗长度）
    # text_splitter = CharacterTextSplitter(separator='。', chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=['\n\n', '\n', ' ', '。'])
    split_docs = text_splitter.split_documents(pages)
    print(split_docs)

    # txt_files = find_txt_files(dir_path)
    # for line in get_singel_jsonFile(file_path):
    #     print(line)


