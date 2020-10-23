import nltk

def highlightSentence(title,text,word_list):

    if len(text)==0:
        return ''

    stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]

    new_list = []
    for word in word_list:
        if word.lower() not in stopwords:
            new_list.append(word)
    word_list = new_list
    

    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_tokenizer.tokenize(text)
    
    sentences_word_list = []
    for sentence in sentences:
        sentences_word_list.append(nltk.word_tokenize(sentence))

    scores = []
    for words in sentences_word_list:
        count = 0
        for word in words:

            for keyword in word_list:
                if word.lower() == keyword.lower():
                    count = count+1
            
        scores.append(count)

    html_text='<div> <div style=\'margin-left: 5px; margin-top: 10px;\'>'+title+':</div><div style=\'margin-left: 50px; margin-right: 10px;\'>'
    for i in range(0,len(scores)):
        class_txt = ''
        if scores[i]<8:
            class_txt = 'normal_text'
        if scores[i]>=8:
            class_txt = 'suspicious_text'
        
        html_text = html_text+'<span class=\''+class_txt+'\'>'+sentences[i]+'</span>'

    html_text=html_text+'</div></div>'

    return html_text

