import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stop(word_list):
    return [w for w in word_list if w not in stop_words]

def clean(line):
    line = line.lower()
    line = line.strip()
    line = re.sub(r'\[(.*?)\]', '', line)
    line = re.sub(r'\((.*?)\)', '', line)
    line = line.replace('\u2005', ' ')
    line = line.replace('\u205f', ' ')
    line = line.replace("-", ' ')

    def remove_chars(input, except_):
        return ''.join([c for c in input if c in except_])

    line = remove_chars(line, [c for c in "abcdefghijklmnopqrstuvwxyz1234567890' "])

    return line

