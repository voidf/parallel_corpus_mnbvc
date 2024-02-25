from datasets import load_dataset
import easyocr

reader = easyocr.Reader(['ar'])

res = reader.readtext(r'C:\Users\Administrator\Documents\parallel_corpus_mnbvc\alignment\ara.png')
print(res)