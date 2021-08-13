import string
import numpy as np
import re

without_stopwords = ['aef', '2fd', 'ccc']
pattern = re.compile('[0-9]+')
cleaned_text = [w for w in without_stopwords if not pattern.findall(w)]
print(cleaned_text)
