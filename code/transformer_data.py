#this is going to create a different embedding

import pandas as pd
def load_data(file, vocab_size, par_size):
    dataset = pd.read_excel(file)
    text = dataset['Text']
    vocab_freq = {}
    for p in text:
        words = p.split()
        for word in words:
            if word in vocab_freq.keys():
                vocab_freq[word] += 1
            else:
                vocab_freq[word] = 1
    vocab_freq = dict(sorted(vocab_freq.items(), key=lambda item: item[1], reverse= True))
    
    count = 1
    vocab_order = {}
    for key in vocab_freq.keys():
        vocab_order[key] = count
        count += 1
        if count > vocab_size:
            break
    print(len(vocab_order.keys()))
    embed_list = []
    for p in text:
        words = p.split()
        if par_size > len(words):
            par_size = len(words)
        embed_word_list = []
        
        for i in range(par_size):
            if words[i] in vocab_order.keys():
                embed_word_list.append(vocab_order[words[i]])
        #print(embed_word_list)
        embed_list.append(embed_word_list)
    return embed_list
    
    

if __name__ == '__main__':
    list1 = load_data("lemmatized_TR.xlsx",4000, 200)
    print(list1)
        

