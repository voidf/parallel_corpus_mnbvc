import datetime
import os

from datasets import load_dataset

from bertalign import Bertalign

langs = ['zh', 'fr', 'es', 'ru']
langs = ['zh']

def main(row, id):
    dst = 'en'
    dst_text = row[dst].replace('\n----\n', '\n')
    # zh = row['zh'].replace('\n----\n', '\n')
    for src in langs:
        src_text = row[src].replace('\n----\n', '\n')
        output_dir = f'aligned/{src}_{dst}/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_filename = f'{output_dir}{id}.txt'

        if os.path.exists(output_filename):
            print('skip', output_filename)
            continue

        aligner = Bertalign(src_text, dst_text, src_lang=src, tgt_lang=dst)
        aligner.align_sents()

        with open(f'{output_filename}', 'w', encoding='utf-8') as f:
            for aligned in aligner.yield_sents():
                f.write(aligned + '=' * 10 + '\n')
    # aligner.
    # aligner.print_sents()


if __name__ == "__main__":
    begin_time = datetime.datetime.now()
    dataset = load_dataset("ranWang/UN_PDF_TEXT_DATA", split='randomTest')
    dataset.map(main, with_indices=True)
    end_time = datetime.datetime.now()
    print('Time elapsed:', end_time - begin_time)