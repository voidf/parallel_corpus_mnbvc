from datasets import load_dataset
import string
import pickle
import jieba
import re
import datetime
import pickle

from transformers import pipeline

langs = ['zh', 'es', 'fr', 'ru', 'ar']
# with open('localtranslation.pkl', 'rb') as f:
    # translators =  pickle.load(f)
translators = {
    # k: pipeline("translation", model=f"Helsinki-NLP/opus-mt-{k}-en") for k in langs
    'zh-en': pipeline("translation", model=f"Helsinki-NLP/opus-mt-zh-en"),
    'en-zh': pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-zh"),
}
# with open('localtranslation.pkl', 'wb') as f:
    # pickle.dump(translators, f)

def translate(source_text: str, source_lang='zh') -> str:
    return translators[source_lang](source_text)['translation_text']

if __name__ == "__main__":
    now_timer = datetime.datetime.now()
    src = '''United Nations E/CN.7/2003/10
Economic and Social Council Distr.: General
29 January 2003
Original: English
V.03-80680 (E) 270203 280203
*0380680*
Commission on Narcotic Drugs
Forty-sixth session
Vienna, 8-17 April 2003
Item 5 of the provisional agenda*
Illicit drug traffic and supply
Strengthening international cooperation in the control of
opium poppy cultivation
Report of the Executive Director
1. In its resolution 45/10, the Commission on Narcotic Drugs called upon the
United Nations International Drug Control Programme (UNDCP) to strengthen its
capacity in Afghanistan in the key thematic areas of drug control so that it could
provide the necessary technical support, subject to the availability of voluntary
resources, to mainstream drug control as a cross-cutting issue in reconstruction and
development, giving priority to areas under opium poppy cultivation, and called
upon the Executive Director to submit to the Commission at its forty-sixth session a
report on the progress made in the implementation of the resolution. The present
report is submitted pursuant to that request.
'''
    ds = load_dataset('bot-yaya/UN_PDF_SUBSET_PREPROCESSED', split='train')
    result = translators['en-zh'](src)
    result = result[0]['translation_text']
    print(result)
    result = translators['zh-en'](result)[0]['translation_text']
    print(result)

    print('running time:', (datetime.datetime.now() - now_timer).total_seconds())
