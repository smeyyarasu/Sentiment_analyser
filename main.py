import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd


nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')


# Load spaCy large transformer model 
nlp = spacy.load('en_core_web_trf')
sentiment_analyser = SentimentIntensityAnalyzer()

def preprocessor(comment):
    comment = comment.lower()
    doc = nlp(comment)
    tokens = []
    nouns = []
    for token in doc:
        if not (token.is_stop or token.like_num or token.is_punct or token.is_space or token.like_url or len(token) == 1):
            tokens.append(token.lemma_)
        if token.pos_ == "NOUN":
            nouns.append(token.lemma_)
    return ' '.join(tokens), ' '.join(nouns)

def scorer(comments):
    scores = []
    pos_comments = []
    neg_comments = []
    for comment in comments:
        processed_comment, nouns = preprocessor(comment)
        score = sentiment_analyser.polarity_scores(processed_comment)
        scores.append(score)
        if score["compound"] > 0.2:
            pos_comments.append(nouns)
        elif score["compound"] < -0.2:
            neg_comments.append(nouns)
    pos_comments = " ".join(pos_comments)
    neg_comments = " ".join(neg_comments)
    return scores, pos_comments, neg_comments

def compute_score(comments):
    if len(comments) == 0:
        return "NA", [], []

    scores, pos_comments, neg_comments = scorer(comments)
    
    # Compute averages
    df = pd.DataFrame(scores)
    averages = df.mean()
    neu = averages["neu"] / 2
    pos = averages["pos"]

    
    score = neu + pos
    score = round(score * 100)
    
    # Extract features frequency
    pos_frequency = nltk.FreqDist(nltk.tokenize.word_tokenize(pos_comments))
    neg_frequency = nltk.FreqDist(nltk.tokenize.word_tokenize(neg_comments))
    
    pos_most_common = pos_frequency.most_common(10)
    neg_most_common = neg_frequency.most_common(10)
    
    pos_ordered_common_features = []
    dict_pos_features = {}
    for i in pos_most_common:
        pos_ordered_common_features.append(i[0])
        dict_pos_features[i[0]] = i[1]
    
    neg_ordered_common_features = []
    dict_neg_features = {}
    for i in neg_most_common:
        neg_ordered_common_features.append(i[0])
        dict_neg_features[i[0]] = i[1]
    
    pos_fea = []
    neg_fea = []
    pos_count = 0
    neg_count = 0
    
    for i in pos_ordered_common_features:
        if pos_count >= 2:
            break
        if i not in dict_neg_features or (dict_pos_features[i] > dict_neg_features[i]):
            pos_fea.append(i)
            pos_count += 1
    
    for i in neg_ordered_common_features:
        if neg_count >= 2:
            break
        if i not in dict_pos_features or (dict_neg_features[i] >= dict_pos_features[i]):
            neg_fea.append(i)
            neg_count += 1

    return score, pos_fea, neg_fea

print("Enter your comments one by one. Type DONE when finished:")
comments = []
while True:
    c = input()
    if c.strip().upper() == "DONE":
        break
    comments.append(c.strip())
    
score, pos_features, neg_features = compute_score(comments)
    
print(f"\nOverall Sentiment Score: {score}")
print(f"Top Positive Features: {pos_features}")
print(f"Top Negative Features: {neg_features}")
