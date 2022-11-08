from matplotlib.colors import is_color_like
import webcolors

class ColorHelper(object):
  def __init__(self):
    pass

  def naive_preprocess_text(self, text):
    text = text.replace('grey', 'gray')
    return text

  def check_color_adv(self, word):
    return word in ['light', 'dark']

  def extract_color(self, text):
    text = self.naive_preprocess_text(text)
    list_words = text.split()
    res = []
    for i, word in enumerate(list_words):
      if is_color_like(word):
        tmp = {'color': word, 'adv': None}
        if (f'{list_words[i-1]}{word}' in webcolors.CSS3_NAMES_TO_HEX 
              or self.check_color_adv(list_words[i-1])):
          tmp['adv'] = list_words[i-1] 
        res.append(tmp)
    return res