########### GUI IMPORTS ################
import streamlit as st
import ssl
############# Displaying images on the front end #################
st.set_page_config(page_title="Summarize and Talk ot your Text",
                   page_icon='📖',
                   layout="centered",  #or wide
                   initial_sidebar_state="expanded",
                   menu_items={
                        'Get Help': 'https://docs.streamlit.io/library/api-reference',
                        'Report a bug': "https://www.extremelycoolapp.com/bug",
                        'About': "# This is a header. This is an *extremely* cool app!"
                                },
                   )
########### SSL FOR PROXY ##############
ssl._create_default_https_context = ssl._create_unverified_context

#### IMPORTS FOR AI PIPELINES ###############
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

from transformers import AutoModel, T5Tokenizer, T5Model
from transformers import T5ForConditionalGeneration
from langchain.llms import HuggingFacePipeline
import torch
import datetime

#############################################################################
#               SIMPLE TEXT2TEXT GENERATION INFERENCE
#           checkpoint = "./models/LaMini-Flan-T5-783M.bin" 
# ###########################################################################
checkpoint = "./model/"  #it is actually LaMini-Flan-T5-248M
LaMini = './model/'

######################################################################
#     SUMMARIZATION FROM TEXT STRING WITH HUGGINGFACE PIPELINE       #
######################################################################
def AI_SummaryPL(checkpoint, text, chunks, overlap):

    """
    checkpoint is in the format of relative path
    example:  checkpoint = "/content/model/"  #it is actually LaMini-Flan-T5-248M   #tested fine
    text it is either a long string or a input long string or a loaded document into string
    chunks: integer, lenght of the chunks splitting
    ovelap: integer, overlap for cor attention and focus retreival
    RETURNS full_summary (str), delta(str) and reduction(str)

    post_summary14 = AI_SummaryPL(LaMini,doc2,3700,500)
    USAGE EXAMPLE:
    post_summary, post_time, post_percentage = AI_SummaryPL(LaMini,originalText,3700,500)
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = chunks,
        chunk_overlap  = overlap,
        length_function = len,
    )
    texts = text_splitter.split_text(text)
    #checkpoint = "/content/model/"  #it is actually LaMini-Flan-T5-248M   #tested fine
    checkpoint = checkpoint
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    base_model = T5ForConditionalGeneration.from_pretrained(checkpoint,
                                                        device_map='auto',
                                                        torch_dtype=torch.float32)
    ### INITIALIZING PIPELINE
    pipe_sum = pipeline('summarization', 
                        model = base_model,
                        tokenizer = tokenizer,
                        max_length = 350, 
                        min_length = 25
                        )
    ## START TIMER
    start = datetime.datetime.now() #not used now but useful
    ## START CHUNKING
    full_summary = ''
    for cnk in range(len(texts)):
      result = pipe_sum(texts[cnk])
      full_summary = full_summary + ' '+ result[0]['summary_text']
    stop = datetime.datetime.now() #not used now but useful  
    ## TIMER STOPPED AND RETURN DURATION
    delta = stop-start  
    ### Calculating Summarization PERCENTAGE
    reduction = '{:.1%}'.format(len(full_summary)/len(text))
    print(f"Completed in {delta}")
    print(f"Reduction percentage: ", reduction)
    
    return full_summary, delta, reduction


global text_summary

### HEADER section
st.image('Headline-text.jpg', width=750)
title = st.text_area('Insert here your Copy/Paste text', "", height = 350, key = 'copypaste')
btt = st.button("1. Start Summarization")
txt = st.empty()
timedelta = st.empty()
text_lenght = st.empty()
redux_bar = st.empty()
st.divider()
down_title = st.empty()
down_btn = st.button('2. Download Summarization') 
text_summary = ''

def start_sum(text):
    if st.session_state.copypaste == "":
        st.warning('You need to paste some text...', icon="⚠️")
    else:
        with st.spinner('Initializing pipelines...'):
            st.success(' AI process started', icon="🤖")
            print("Starting AI pipelines")
            text_summary, duration, reduction = AI_SummaryPL(LaMini,text,3700,500)
        txt.text_area('Summarized text', text_summary, height = 350, key='final')
        timedelta.write(f'Completed in {duration}')
        text_lenght.markdown(f"Initial length = {len(text.split(' '))} words / summarization = **{len(text_summary.split(' '))} words**")
        redux_bar.progress(len(text_summary)/len(text), f'Reduction: **{reduction}**')
        down_title.markdown(f"## Download your text Summarization")



if btt:
    start_sum(st.session_state.copypaste)

if down_btn:
    def savefile(generated_summary, filename):
        st.write("Download in progress...")
        with open(filename, 'w') as t:
            t.write(generated_summary)
        t.close()
        st.success(f'AI Summarization saved in {filename}', icon="✅")
    savefile(st.session_state.final, 'text_summarization.txt')
    txt.text_area('Summarized text', st.session_state.final, height = 350)



