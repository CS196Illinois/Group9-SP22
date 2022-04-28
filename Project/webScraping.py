from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def getTitle(link):
    html_txt = requests.get(link).text
    soup2 = BeautifulSoup(html_txt, 'lxml')
    title_element = soup2.find_all('h1', class_ = 'ArticleHeader-headline')[0].text
    return title_element

def getTime(link):
    html_txt = requests.get(link).text
    soup2 = BeautifulSoup(html_txt, 'lxml')
    timeElement = [x.text.strip() for x in soup2.find_all('time')]
    return timeElement

def getMainText(link):
    html_txt = requests.get(link).text
    soup2 = BeautifulSoup(html_txt, 'lxml')
    text2 = soup2.find_all('p')
    sentenceList = []
    for x in range(len(text2)):
        temptxt2 = text2[x].text
        if (len(str(temptxt2)) > 0):
            sentenceList.append(str(temptxt2))
    print(len(sentenceList))
    return sentenceList



sentence1 = getMainText('https://www.cnbc.com/2022/02/25/gas-prices-spike-amid-russia-ukraine-conflict-how-to-save-at-pump.html')
sentence2 = getMainText('https://www.cnbc.com/2022/03/29/yen-on-the-ropes-as-bank-of-japan-defends-yield-target-.html')
embeddings1 = model.encode(sentence1, convert_to_tensor=True)
embeddings2 = model.encode(sentence2, convert_to_tensor=True)
cosine_scores = util.cos_sim(embeddings1, embeddings2)


print(getTitle('https://www.cnbc.com/2022/02/25/gas-prices-spike-amid-russia-ukraine-conflict-how-to-save-at-pump.html'))
print(getTime('https://www.cnbc.com/2022/02/25/gas-prices-spike-amid-russia-ukraine-conflict-how-to-save-at-pump.html'))
print(getMainText('https://www.cnbc.com/2022/02/25/gas-prices-spike-amid-russia-ukraine-conflict-how-to-save-at-pump.html'))

print(getTitle('https://www.cnbc.com/2022/03/29/yen-on-the-ropes-as-bank-of-japan-defends-yield-target-.html'))

print(cosine_scores)
# for next meeting, make sure to implement the NLTK but in order to do this I need to turn the paragraphs of text into a list storing each sentence separated by \ or not
# when you want to add to a list use .append()


# TEST LINK: https://www.cnbc.com/2022/03/29/yen-on-the-ropes-as-bank-of-japan-defends-yield-target-.html
# TEST LINK: https://www.cnbc.com/2022/02/25/gas-prices-spike-amid-russia-ukraine-conflict-how-to-save-at-pump.html
# TEST LINK (Apple stock article): https://www.cnbc.com/2022/04/28/apple-aapl-earnings-q2-2022.html
