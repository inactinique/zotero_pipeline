import re
import numpy as np
from typing import List, Tuple, Dict
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import pyLDAvis
import pyLDAvis.gensim_models
import nltk
from nltk.stem import WordNetLemmatizer
from IPython.display import display, HTML

class LDAProcessor:
    def __init__(self, num_topics=5, num_words=10):
        """
        Initialize the LDA processor
        
        Args:
            num_topics (int): Number of topics to extract
            num_words (int): Number of words per topic
        """
        self.num_topics = num_topics
        self.num_words = num_words
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess the text: tokenize, remove stopwords, lemmatize
        """
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stopwords
        tokens = simple_preprocess(text, deacc=True)
        tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 3]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens

    def perform_lda(self, texts: List[str], metadata: List[Dict]) -> Tuple[LdaModel, corpora.Dictionary, List[List[float]], pd.DataFrame]:
        """
        Perform LDA analysis on the texts
        """
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        
        # Train LDA model
        lda_model = LdaModel(
            corpus=corpus,
            num_topics=self.num_topics,
            id2word=dictionary,
            random_state=42,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Get document-topic matrix
        doc_topics = []
        for doc in corpus:
            topic_probs = [0] * self.num_topics
            topics = lda_model.get_document_topics(doc)
            for topic, prob in topics:
                topic_probs[topic] = prob
            doc_topics.append(topic_probs)
            
        # Create document details DataFrame
        doc_details = []
        for idx, (probs, meta) in enumerate(zip(doc_topics, metadata)):
            dominant_topic = np.argmax(probs)
            doc_details.append({
                'Document': meta.get('title', f'Doc_{idx}'),
                'Dominant_Topic': dominant_topic,
                'Topic_Probability': probs[dominant_topic],
                'Authors': '; '.join(author.get('name', '') for author in meta.get('creators', [])),
                'Date': meta.get('date', ''),
                'Topic_Distribution': probs
            })
        
        doc_details_df = pd.DataFrame(doc_details)
        
        return lda_model, dictionary, corpus, doc_details_df

    def display_topics(self, lda_model: LdaModel):
        """
        Display topics and their top words in a formatted table
        """
        # Create HTML for topics table
        html = """
        <h3>Topic Words</h3>
        <table style='width:100%; border-collapse: collapse;'>
        <tr>
            <th style='border: 1px solid black; padding: 8px;'>Topic</th>
            <th style='border: 1px solid black; padding: 8px;'>Top Words</th>
        </tr>
        """
        
        for topic_id in range(self.num_topics):
            words = lda_model.show_topic(topic_id, self.num_words)
            word_str = ", ".join([f"{word} ({prob:.3f})" for word, prob in words])
            html += f"""
            <tr>
                <td style='border: 1px solid black; padding: 8px;'>Topic {topic_id}</td>
                <td style='border: 1px solid black; padding: 8px;'>{word_str}</td>
            </tr>
            """
        
        html += "</table>"
        display(HTML(html))

    def display_document_topics(self, doc_details_df: pd.DataFrame):
        """
        Display document-topic distribution in a formatted table
        """
        # Style the dataframe
        styled_df = doc_details_df[['Document', 'Dominant_Topic', 'Topic_Probability', 'Authors', 'Date']].style\
            .format({'Topic_Probability': '{:.3f}'})\
            .set_properties(**{'border': '1px solid black',
                             'padding': '8px',
                             'text-align': 'left'})\
            .set_table_styles([{'selector': 'th',
                               'props': [('border', '1px solid black'),
                                       ('padding', '8px'),
                                       ('text-align', 'left')]}])
        
        display(HTML("<h3>Document Topics</h3>"))
        display(styled_df)

    def display_visualization(self, lda_model: LdaModel, corpus: List[List[Tuple]], dictionary: corpora.Dictionary):
        """
        Display interactive visualization in the notebook
        """
        # Prepare visualization
        vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        
        # Display in notebook
        return pyLDAvis.display(vis_data)