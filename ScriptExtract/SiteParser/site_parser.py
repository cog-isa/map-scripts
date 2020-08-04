from bs4 import BeautifulSoup
import requests
import re

class Parser:
    def __init__(self, url=None, keys = None):
        if not url:
            self.url = 'https://ru.wiktionary.org/wiki/'
            self.key_words = ["Значение", "Синонимы", "Антонимы", "Гиперонимы", "Гипонимы"]
        else:
            self.url = url
            self.key_words = keys

    def get_word_info(self, word):
        word_descr = {}
        if ' ' in word:
            word = '_'.join(word.strip().split(' '))
        link = self.url + word
        page = requests.get(link)
        soup = BeautifulSoup(page.text, 'html.parser')
        word_page = soup.find(class_='mw-parser-output')
        word_content = [el for el in word_page.contents if el != '\n']
        descr_len = len(word_content)
        for id, child in enumerate(word_content):
            key = None
            for el in self.key_words:
                try:
                    if el in child.text:
                        key = el
                        break
                except AttributeError:
                    break
            if key:
                word_descr[key] = None
                if id+1 < descr_len:
                    if word_content[id+1].name == 'ol':
                        if key != 'Значение':
                            wt = re.sub(r'\n', ', ', word_content[id+1].text)
                            wt = re.sub(r'\[\w*\]', '', wt)
                            wt = re.sub(r'\([^)]*\)', '', wt)
                            wt = wt.split(', ')
                            wt = [el for el in wt if re.findall('\w+', el)]
                            if wt:
                                word_descr[key] = [el.strip() for el in wt]
                        else:
                            wt = re.sub(r'\n', ' ', word_content[id+1].text)
                            word_descr[key] = wt

        return word_descr


if __name__ == "__main__":
    parser = Parser()
    response0 = parser.get_word_info('легковой автомобиль')
    print(response0)
    response1 = parser.get_word_info('голос')
    response2 = parser.get_word_info('грузовой автомобиль')
    print()
