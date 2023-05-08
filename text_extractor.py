import numpy as np
import pandas as pd
import string
import re
import cv2
import PIL
import pytesseract

import os
import json
from glob import glob
from tqdm import tqdm

import glog as log
import argparse
import tkinter as tk
from tkinter import filedialog
import pickle

import warnings
warnings.filterwarnings('ignore')

def main(config_path):
    log.info("starting main...")
    
    with open(config_path, "r") as c:
        config = json.loads(c.read())
    log.info("config successfully imported")
    
    # prepare data
    image_path = config['image']['image_path']
    imgPaths = glob(image_path)

    log.info("extracting text data from all images in path...")
    extracted_image_data = extract_text_from_img(imgPaths)

    # export untagged data
    log.info("exporting extracted text data...")
    extracted_image_data.to_csv('output/raw_extracted_data.csv',index=False)

    # import tagged data
    log.info("importing tagged text data...please select your tagged/labelled txt data")
    tagged_extracted_data = import_tagged_data()

    # clean tagged data
    log.info("cleaning tagged/labelled data...")
    punctuation = config['cleaning']['punctuation']
    cleaned_tagged_data = clean_tagged_data(tagged_extracted_data,punctuation)

    # export clean tagged data
    log.info("exporting cleaned tagged text data...")
    cleaned_tagged_data.to_csv('output/cleaned_tagged_data.csv',index=False)

    # export data in spacy format
    log.info('converting and exporting data into spacy format...')
    data_spacy_format = convert_to_spacy(cleaned_tagged_data)
    pickle.dump(data_spacy_format,open('output/data_spacy_format.pickle',mode='wb'))

    log.info('done')

def extract_text_from_img(imgPaths):
    allBusinessCard = pd.DataFrame(columns=['id','text'])

    for imgPath in tqdm(imgPaths, desc='Text Extractor'):
        _, filename = os.path.split(imgPath)

        # open image with opencv
        img_cv = cv2.imread(imgPath)

        # extract data from image with pytesseract
        data = pytesseract.image_to_data(img_cv)

        # store data in df
        dataList = list(map(lambda x: x.split('\t'), data.split('\n')))
        df = pd.DataFrame(dataList[1:], columns=dataList[0])
        df.dropna(inplace=True)

        # extract useful data
        df.conf = df.conf.astype(int)
        useFulData = df.query('conf >= 30')
        businessCard = pd.DataFrame()
        businessCard['text'] = useFulData['text']
        businessCard['id'] = filename
        
        # concat all data from images
        allBusinessCard = pd.concat((allBusinessCard,businessCard))
    
    return allBusinessCard

def import_tagged_data():
    root = tk.Tk()
    root.withdraw()
    tagged_data = filedialog.askopenfilename()
    log.info(f'tagged data: {tagged_data}')

    if os.path.exists(tagged_data):
        with open(tagged_data, 'r', encoding='utf8', errors='ignore') as f:
            tagged_extracted_text = f.read()

    return tagged_extracted_text

def clean_tagged_data(text,punctuation):
    data = list(map(lambda x: x.split('\t'),text.split('\n')))
    df = pd.DataFrame(data[1:],columns=data[0])
    df.dropna(inplace=True)    
    df['text'] = df['text'].apply(lambda x: cleanText(x,punctuation))

    cleaned_tagged_data = df.query('text !=""')
    cleaned_tagged_data.dropna(inplace=True)

    return cleaned_tagged_data

def cleanText(txt, punctuation):
    whitespace = string.whitespace
    tableWhitespace = str.maketrans('','',whitespace)
    tablePunctuation = str.maketrans('','',punctuation)
    text = str(txt)
    text = text.lower()

    #remove whitespace
    removewhitespace = text.translate(tableWhitespace)
    #remove punctuations
    removepunctuation = removewhitespace.translate(tablePunctuation)
    
    return str(removepunctuation)

def convert_to_spacy(data):
    allCardsData = []
    group = data.groupby(by='id')
    items = group.groups.keys()
    for item in items:
        grouparray = group.get_group(item)[['text','tag']].values
        content=''
        annotations = {'entities':[]}
        start = 0
        end = 0
        for text,label in grouparray:
            text = str(text)
            stringLenght = len(text) + 1

            start = end
            end = start + stringLenght

            if label != 'O':
                annot = (start,end-1,label)
                annotations['entities'].append(annot)

            content = content + text + ' '

        cardData = (content,annotations)
        allCardsData.append(cardData)
    
    return allCardsData

if __name__ == "__main__": 
       
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path",help="path to config")

    args = parser.parse_args()
    main(**vars(args))