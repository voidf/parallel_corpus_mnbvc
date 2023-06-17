import argparse
import datasets
import json
import time
import random
import os
from filelock import FileLock
import wandb
from pathlib import Path
import sys

# 防止用户在其他位置调用此脚本，从而找不到库的情况
current_dir = os.path.dirname(os.path.abspath(__file__))
alignment_path = os.path.join(current_dir, '..')
sys.path.append(alignment_path)

import alignment.utils as utils
from alignment.batch_detector import GPTBatchDetector


# 所有的缓存与相关文件同意放到标本脚本目录处
RECORD_INDEX_MAP_FILE_LOACATION = f"{os.path.dirname(os.path.abspath(__file__))}/record_index_map.json"
LOCK_FILE_LOCATION = f"{os.path.dirname(os.path.abspath(__file__))}/record_index_map.json.lock"
CACHE_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/gpt_cache"

Path(CACHE_DIR).mkdir(exist_ok=True)

class ParagraphAssembler:

    def __init__(self, test=False):
        self.test = test
        self.dataset_row = self.get_dataset_row()
        print(f"{self.record} start")

        if test:
            self.done_in_json_settings_file()
            print(f"{self.record} success")
        

    def done_in_json_settings_file(self):
        """
        In the json file set the current file is done

        """

        # lock 可初步理解成自旋锁
        with FileLock(LOCK_FILE_LOCATION):
            with open(RECORD_INDEX_MAP_FILE_LOACATION, "r+") as f:
                record_index_map = json.load(f)
                record_index_map[self.record]["completed"] = True # record已完成
                record_index_map[self.record]["processing"] = False # record处理完毕，所以为Flase

                f.seek(0) # 将文件指针移动到文件开头
                f.truncate()  # 清空文件内容

                json.dump(record_index_map, f) # 将更新后的record索引映射写回文件
                f.flush() # 将文件内容刷新到磁盘

    def get_dataset_row(self):
        """
        Get unused rows in datasets

        Returns: 
            record: file record number
            dataset_row: dict('zh', 'en', 'fr', 'es', 'ru', 'record')

        """

        # lock 可初步理解成自旋锁
        with FileLock(LOCK_FILE_LOCATION):
            with open(RECORD_INDEX_MAP_FILE_LOACATION, "r+") as f:
                record_index_map = json.load(f)
                
                # 准备的数据集索引
                prepare_dataset_index = None 

                # 寻找下一个可用的record，并找到对应dataset的index
                for record in record_index_map:
                    completed = record_index_map[record]["completed"] # record是否已完成
                    processing = record_index_map[record]["processing"] # record是否正在处理

                    # 如果record既未完成也未在处理中
                    if not (completed or processing):
                        prepare_dataset_index = record_index_map[record]["index"]
                        record_index_map[record]["processing"] = True
                        self.record = record
                        break

                f.seek(0) # 将文件指针移动到文件开头
                f.truncate() # 清空文件内容

                json.dump(record_index_map, f) # 将更新后的record索引映射写回文件
                f.flush() # 将文件内容刷新到磁盘

                dataset = datasets.load_dataset("ranWang/un_pdf_text_data_test", verification_mode="no_checks")["randomTest10000"]
                # 返回准备的数据集索引对应的数据
                return dataset[prepare_dataset_index]



    def start(self):
        detector = GPTBatchDetector('gpt-remote', CACHE_DIR)
        lines = self.dataset_row['en'].splitlines()

        predicted = detector.detect(lines, record_id=self.record)
        self.predicted = predicted
        return self.predicted


    def post_process(self):
        pass


    def batch_post_process(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, default=False, help='openai api key')
    parser.add_argument('--test', type=str, default=False, help='是否测试此脚本')

    args = parser.parse_args()
    key = args.key

    if not key:
        raise ValueError("params --key must input")

    os.environ['OPENAI_API_KEY'] = key

    paragraphAssembler = ParagraphAssembler(args.test)

    wandb.init(project="paragraph_assembler", name=f"GPTBatchDetector-{paragraphAssembler.record}")
    run = wandb.run

    artifact = wandb.Artifact(
        name="paragraph_assembler",
        type="dataset",
        description="JSON files only containing predictions and record_id",
        metadata=dict(record=paragraphAssembler.record))


    paragraphAssembler.start()
    paragraphAssembler.post_process()

    with artifact.new_file(f"{paragraphAssembler.record}-is-hard-linebreak.json", mode="w") as f:
        json.dump(paragraphAssembler.predicted, f)

    run.log_artifact(artifact)

    wandb.finish()
    
