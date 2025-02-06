import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup
import jieba
from urllib.parse import urljoin

jieba.setLogLevel(logging.INFO)
MAX_ARTICLES_PER_SITE = 30

# Step 1: Fetch the main page and extract article links
def get_article_links(topic_url):
    response = requests.get(topic_url)
    response.encoding = 'utf-8'  # Ensure correct encoding for Chinese characters
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the div with class="yaowen"
    yaowen_div = soup.find('div', class_='yaowen')
    if not yaowen_div:
        print("No div with class 'yaowen' found.")
        return []

    pattern = re.compile(r'https?://finance\.caixin\.com/\d{4}-\d{2}-\d{2}/.*\.html')
    # Find all <a> tags inside the yaowen div
    article_links = set()  # Use a set to ensure uniqueness
    for link in yaowen_div.find_all('a', href=True):
        href = link['href']
        full_url = urljoin(topic_url, href)  # Convert relative URLs to absolute
        if pattern.match(full_url):
            article_links.add(full_url)  # Add to the set

    return list(article_links)  # Convert set back to list

# Step 2: Fetch and extract text from a single article
def get_article_text(article_url):
    response = requests.get(article_url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract text from the article (adjust the selector based on the website's structure)
    article_text = ''
    for paragraph in soup.find_all('p'):  # Assuming text is in <p> tags
        article_text += paragraph.get_text() + '\n'

    return article_text

def load_stop_words():
    stop_words_files = ["baidu_stopwords.txt", "cn_stopwords.txt", "hit_stopwords.txt", "scu_stopwords.txt"]
    stop_words = set()
    for file_name in stop_words_files:
        file_path = Path("stopwords") / file_name
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                stop_words.update(file.read().splitlines())  # Add stop words to the set
        except FileNotFoundError:
            print(f"Warning: Stop words file '{file_name}' not found in '{stop_words_dir}'.")
        except Exception as e:
            print(f"Error reading '{file_name}': {e}")
    return stop_words

def filter_tokens(tokens, stop_words):
    chinese_pattern = re.compile(r'^[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df]+$')
    filtered = [word for word in tokens if (word not in stop_words and chinese_pattern.match(word))]
    pattern = re.compile(r'.*[a-zA-Z0-9].*')
    return [word for word in filtered if not pattern.match(word)]

# Main function to crawl articles and process text
def crawl_and_process(topic_url: str, stop_words: list[str]):
    # Get all article links
    print("-"*100)
    print(f"Processing: {topic_url}")
    article_links = get_article_links(topic_url)
    print(f"Found {len(article_links)} articles.")
    print(f"Only processing first {MAX_ARTICLES_PER_SITE} articles.")
    # print("\n".join(article_links))

    filtered_documents = []

    # Process each article
    for i, link in enumerate(article_links):
        if i == MAX_ARTICLES_PER_SITE:
            break
        print(f"Processing article {i + 1}: {link}")
        try:
            # Fetch article text
            text = get_article_text(link)
            # Tokenize the text
            tokens = jieba.lcut(text)
            # Filter the tokens
            filtered_tokens = filter_tokens(tokens, stop_words)
            filtered_documents.append(filtered_tokens)
            # Print or save the tokens
            # print(f"Filtered tokens: {filtered_tokens[:50]}...")  # Print first 50 tokens as a sample
        except Exception as e:
            print(f"Error processing {link}: {e}")

    tfidf_array, feature_names = apply_tfidf(filtered_documents)

    # Analyze the TF-IDF results
    for i, doc_tfidf in enumerate(tfidf_array):
        print(f"\nTop tokens for article {i + 1} ({article_links[i]}):")
        # Get the indices of the top N tokens with the highest TF-IDF scores
        top_n = 10
        top_indices = doc_tfidf.argsort()[-top_n:][::-1]
        for idx in top_indices:
            print(f"{feature_names[idx]}: {doc_tfidf[idx]:.4f}")

    km = KMeans(n_clusters=5, random_state=0).fit(tfidf_array)
    for c in range(5):
        docs_in_clust = [i for i, clust in enumerate(km.labels_)]
        print(f"cluster {c}: {docs_in_clust}")



# Step 6: Apply TF-IDF to the filtered tokens
def apply_tfidf(filtered_documents):
    # Join the tokens into a single string for each document
    documents = [" ".join(tokens) for tokens in filtered_documents]

    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the documents into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Get the feature names (tokens)
    feature_names = vectorizer.get_feature_names_out()

    # Convert the TF-IDF matrix to an array for easier analysis
    tfidf_array = tfidf_matrix.toarray()

    return tfidf_array, feature_names

# Run the scraper
if __name__ == "__main__":
    topics = ["finance"]#, "companies", "international"]
    stop_words = load_stop_words()
    for topic in topics:
        crawl_and_process(f"https://{topic}.caixin.com/", stop_words)
