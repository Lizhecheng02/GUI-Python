import streamlit as st
import os
import spacy
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from spacy.lang.en import English


def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer('english'))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result


# @st.cache_data
def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(str(my_text))
    all_data = [('"Token": {} , \n "Lemma": {}'.format(
        token.text, token.lemma_)) for token in docx]
    return all_data


# @st.cache_data
def entity_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(str(my_text))
    tokens = [token.text for token in docx]
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    all_data = ['"Token": {} , \n "Entities": {}'.format(tokens, entities)]
    return all_data


def main():
    st.title('NLPiffy with Streamlit')
    st.subheader('Natural Language Processing On the Go...')
    st.markdown(
        """
    	#### Description
    	+ This is a Natural Language Processing(NLP) Based App useful for basic NLP task
    	Tokenization, NER, Sentiment, Summarization
    	"""
    )

    if st.checkbox('Show Tokens and Lemma'):
        st.subheader('Tokenize your text')
        message = st.text_area('Enter Text', 'Type Here ...', key=0)
        if st.button('Analyze'):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)

    if st.checkbox('Show Named Entities'):
        st.subheader('Analyze your text')
        message = st.text_area('Enter text', 'Type Here ...', key=1)
        if st.button('Extract'):
            entity_result = entity_analyzer(message)
            st.json(entity_result)

    if st.checkbox('Show sentiment analysis'):
        st.subheader('Analyze your text')
        message = st.text_area('Enter text', 'Type Here ...', key=2)
        if st.button('Analyze Sentiment'):
            blob = TextBlob(message)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)

    if st.checkbox('Show text summarization'):
        st.subheader('Summarize your text')
        message = st.text_area('Enter text', 'Type Here ...', key=3)
        summary_options = st.selectbox('Choose Summarizer', ['sumy'])
        if st.button('Summarize'):
            if summary_options == 'sumy':
                st.text('Using Sumy Summarizer ...')
                summary_result = sumy_summarizer(message)
            st.success(summary_result)

    st.sidebar.subheader('About App')
    st.sidebar.text('NLPiffy App with Streamlit')
    st.sidebar.info('Cudos to the Streamlit Team')

    st.sidebar.subheader('By')
    st.sidebar.text('Zhecheng Li')


if __name__ == '__main__':
    main()
