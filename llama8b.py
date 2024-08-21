#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import pandas as pd
import numpy as np

import json
import os
import unicodedata

import random
# from transformers import set_seed

# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)
# set_seed(seed)

from tqdm import tqdm
import pymupdf
import pymupdf4llm
from collections import Counter

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

from accelerate import Accelerator

import langchain

from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter, MarkdownHeaderTextSplitter
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain.document_loaders import PDFPlumberLoader, PyMuPDFLoader, PyPDFLoader, UnstructuredPDFLoader

import peft
from peft import PeftModel

import datasets
from datasets import Dataset
from transformers import Trainer, TrainingArguments


# In[2]:


import transformers


# In[3]:


from sentence_transformers import SentenceTransformer, util


# In[4]:


class Opt:
    def __init__(self):
        self.model_configs = {
            'meta-llama/Meta-Llama-3.1-8B-Instruct':
            {
                'quantization_config': None,
                'torch_dtype': 'auto',
                'max_token': 256,
            },
            
            'rtzr/ko-gemma-2-9b-it':{
                'quantization_config': BitsAndBytesConfig(
                    load_in_4bit= True,
                    bnb_4bit_use_double_quant= True,
                    bnb_4bit_quant_type= 'nf4',
                    bnb_4bit_compute_dtype= torch.bfloat16
                ),
                'torch_dtype': 'auto',
                'max_token': 512
            }
        }

        self.llm_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.llm_model_config = self.model_configs[self.llm_model]
        self.llm_peft = False
        self.llm_peft_checkpoint = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        # self.embed_models = ['distilbert-base-uncased', 'intfloat/multilingual-e5-large']
        self.embed_models = ["intfloat/multilingual-e5-base", "jhgan/ko-sbert-nli", "intfloat/multilingual-e5-large"]
        self.embed_model = self.embed_models[1]

        self.pdf_loader = 'pymupdf'

        self.base_directory = './'
        self.train_csv_path = os.path.join(self.base_directory, 'train.csv')
        self.test_csv_path = os.path.join(self.base_directory, 'test.csv')
        self.chunk_size = 512
        self.chunk_overlap = 32

        self.ensemble = True
        self.bm25_w = 0.5
        self.faiss_w = 0.5

        self.is_submit = True
        self.eval_sum_mode = False

        self.output_dir = 'test_results'
        self.output_csv_file = f"{self.llm_model.split('/')[1]}_{self.embed_model.split('/')[1]}_pdf{self.pdf_loader}_chks{self.chunk_size}_chkovp{self.chunk_overlap}_bm25{self.bm25_w}_faiss{self.faiss_w}_mix_submission.csv"
        os.makedirs(self.output_dir, exist_ok=True)

    def to_json(self):
        return json.dumps(self.__dict__)

args = Opt()


# In[5]:


from huggingface_hub import login
import dotenv
from dotenv import load_dotenv


# In[6]:


load_dotenv()


# In[7]:


os.getenv('token')


# In[8]:


load_dotenv()

token = os.getenv('token')

login(token = token)


# In[9]:


from huggingface_hub import hf_hub_download


# In[10]:


import transformers


# In[11]:


def load_train_data(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
    return train_df[['Question', 'Answer']]


# In[12]:


def train_question_improver(train_df):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    train_embeddings = model.encode(train_df['Question'].tolist(), convert_to_tensor=True)
    return model, train_embeddings


# In[13]:


def improve_question(model, train_embeddings, train_df, question):
    question_embedding = model.encode(question, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(question_embedding, train_embeddings)
    best_index = torch.argmax(cosine_scores).item()
    improved_question = train_df['Question'].iloc[best_index]
    return improved_question


# In[14]:


import accelerate


# In[15]:


from accelerate import init_empty_weights, load_checkpoint_and_dispatch


# In[16]:


import time


# In[17]:


from accelerate import disk_offload


# In[20]:


help(disk_offload)


# In[23]:


def setup_llm_pipeline():
    start_time = time.time()
    print('started: ', start_time)
    
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    tokenizer.use_default_system_prompt = False

    accelerator = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        quantization_config=args.llm_model_config['quantization_config'],
        torch_dtype=args.llm_model_config['torch_dtype'],
        device_map='auto',
        trust_remote_code=True
    )


    # model = disk_offload(
    #     model,
    #     offload_dir='./to/offload_folder/'
    # )
    
    if args.llm_peft:
        model = PeftModel.from_pretrained(model, args.llm_peft_checkpoint)

        
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        return_full_text=False,
        max_new_tokens=args.llm_model_config['max_token']
    )

    end_time = time.time()
    print(f"Model loading time: {end_time - start_time:.2f} seconds")
    
    return HuggingFacePipeline(pipeline=text_generation_pipeline)


# In[24]:


llm = setup_llm_pipeline()


# In[25]:


def process_pdf(file_path):
    md_text = pymupdf4llm.to_markdown(file_path)
    header_split = [
        ('#', 'header I'),
        ('##', 'header II'),
        ('###', 'header III'),
    ]

    md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=header_split, strip_headers=False)
    md_chunks = md_header_splitter.split_text(md_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )

    splits = text_splitter.split_documents(md_chunks)
    return splits


# In[26]:


def create_vector_db(chunks, model_path, method='faiss'):
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    db = FAISS.from_documents(chunks, embedding=embeddings)
    return db


# In[27]:


def process_single_pdf(pdf_path):
    chunks = process_pdf(pdf_path)
    db = create_vector_db(chunks, model_path=args.embed_model)
    
    kiwi_bm25_retriever = KiwiBM25Retriever.from_documents(chunks)
    faiss_retriever = db.as_retriever()

    retriever = EnsembleRetriever(
        retrievers=[kiwi_bm25_retriever, faiss_retriever],
        weights=[args.bm25_w, args.faiss_w],
        search_type='mmr'
    )

    del chunks, db, kiwi_bm25_retriever, faiss_retriever
    torch.cuda.empty_cache()

    return retriever


# In[28]:


def process_questions_for_pdf(pdf_path, questions_df, model, train_embeddings, train_df):
    retriever = process_single_pdf(pdf_path)
    llm_pipeline = setup_llm_pipeline()

    answers = []
    for _, row in questions_df.iterrows():
        question = row['Question']
        print(f"Original question: {question}")
        
        improved_question = improve_question(model, train_embeddings, train_df, question)
        print(f"Improved question: {improved_question}")

        result = llm_pipeline(improved_question)
        
        if isinstance(result, list):
            answer_text = result[0]['generated_text'] if 'generated_text' in result[0] else result[0]
        elif isinstance(result, dict):
            answer_text = result.get('generated_text', result)
        else:
            answer_text = result

        answers.append({
            'SAMPLE_ID': row['SAMPLE_ID'], 
            'Answer': answer_text
        })
    
    del retriever, llm_pipeline
    torch.cuda.empty_cache()

    return answers


# In[29]:


def process_test_questions(df, model, train_embeddings, train_df):
    all_answers = []
    unique_paths = df['Source_path'].unique()

    for path in tqdm(unique_paths, desc='Processing PDFs'):
        pdf_questions_df = df[df['Source_path'] == path]
        answers = process_questions_for_pdf(path, pdf_questions_df, model, train_embeddings, train_df)
        all_answers.extend(answers)

    return all_answers


# In[ ]:


if __name__ == "__main__":
    train_df = load_train_data(args.train_csv_path)
    question_improver_model, train_embeddings = train_question_improver(train_df)
    
    test_df = pd.read_csv(args.test_csv_path)
    answers = process_test_questions(test_df, question_improver_model, train_embeddings, train_df)

    output_path = os.path.join(args.output_dir, args.output_csv_file)
    pd.DataFrame(answers).to_csv(output_path, index=False)


# In[ ]:


print('done')


# In[28]:


output_path


# In[ ]:





# In[51]:


test_res = pd.read_csv("./test_results/skt_kogpt2-base-v2_mobilebert-uncased_pdfpymupdf_chks512_chkovp32_bm250.5_faiss0.5_mix_submission.csv")


# In[52]:


test_res


# In[33]:


test = pd.read_csv("./test.csv")


# In[34]:


test


# In[31]:


train = pd.read_csv("./train.csv")


# In[32]:


train

