#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision
import pandas as pd
import numpy as np

import json
import os
import unicodedata

import random
from transformers import set_seed

seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
set_seed(seed)

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


# In[38]:


class Opt:
    def __init__(self):
        self.model_configs = {
            'skt/kogpt2-base-v2': {
                'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                'torch_dtype': torch.float16,
                'max_token': 512,
            },
        }

        self.llm_model = 'skt/kogpt2-base-v2'
        self.llm_model_config = self.model_configs[self.llm_model]
        self.llm_peft = False
        self.llm_peft_checkpoint = None

        self.embed_models = ['distilbert-base-uncased', 'google/mobilebert-uncased']
        self.embed_model = self.embed_models[1]

        self.pdf_loader = 'pymupdf'

        self.base_directory = './'
        self.train_csv_path = os.path.join(self.base_directory, 'train.csv')
        self.test_csv_path = os.path.join(self.base_directory, 'test.csv')
        self.chunk_size = 256
        self.chunk_overlap = 16

        self.ensemble = True
        self.bm25_w = 0.5
        self.faiss_w = 0.5

        self.is_submit = True
        self.eval_sum_mode = False

        self.output_dir = 'test_results'
        self.output_csv_file = f"{self.llm_model.replace('/', '_')}_{self.embed_model.split('/')[1]}_pdf{self.pdf_loader}_chks{self.chunk_size}_chkovp{self.chunk_overlap}_bm25{self.bm25_w}_faiss{self.faiss_w}_mix_submission.csv"
        os.makedirs(self.output_dir, exist_ok=True)

    def to_json(self):
        return json.dumps(self.__dict__)

args = Opt()


# In[ ]:


def setup_llm_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    tokenizer.use_default_system_prompt = False

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        quantization_config=args.llm_model_config['quantization_config'],
        torch_dtype=args.llm_model_config['torch_dtype'],
        device_map='auto',
        trust_remote_code=True
    )

    if args.llm_peft:
        model = PeftModel.from_pretrained(model, args.llm_peft_checkpoint)

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        return_full_text=False,
        max_new_tokens=args.llm_model_config['max_token']
    )

    return HuggingFacePipeline(pipeline=text_generation_pipeline)


# In[9]:


def normalize_path(path):
    return unicodedata.normalize('NFC', path)

def format_docs(docs):
    return '\n'.join([doc.page_content for doc in docs])

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


# In[ ]:


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

def process_questions_for_pdf(pdf_path, questions_df):

    retriever = process_single_pdf(pdf_path)

    llm_pipeline = setup_llm_pipeline()

    answers = []
    for _, row in questions_df.iterrows():
        question = row['Question']
        print(f"Generating answer for: {question}")

        result = llm_pipeline(question)
        # answers.append({
        #     'ID': row['SAMPLE_ID'],
        #     'Answer': result[0]['generated_text']
        # })

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
    
    # 메모리 해제
    del retriever, llm_pipeline
    torch.cuda.empty_cache()

    return answers


# In[13]:


def process_test_questions(df):

    all_answers = []
    unique_paths = df['Source_path'].unique()

    for path in tqdm(unique_paths, desc='Processing PDFs'):
        pdf_questions_df = df[df['Source_path'] == path]

        answers = process_questions_for_pdf(path, pdf_questions_df)
        all_answers.extend(answers)

    return all_answers


# In[14]:


test = pd.read_csv('./test.csv')


# In[15]:


test


# In[16]:


train = pd.read_csv("./train.csv")
train


# In[17]:


if __name__ == "__main__":
    test_df = pd.read_csv(args.test_csv_path)
    answers = process_test_questions(test_df)
    print(answers)

    # 결과 저장
    output_path = os.path.join(args.output_dir, args.output_csv_file)
    pd.DataFrame(answers).to_csv(output_path, index=False)


# In[18]:


test


# In[24]:


pd.read_csv("./test_results/skt_kogpt2-base-v2_mobilebert-uncased_pdfpymupdf_chks512_chkovp32_bm250.5_faiss0.5_mix_submission.csv")


# In[41]:


test.loc[1, 'Question']


# In[42]:


pd.read_csv("./test_results/skt_kogpt2-base-v2_mobilebert-uncased_pdfpymupdf_chks512_chkovp32_bm250.5_faiss0.5_mix_submission.csv").loc[1, 'Answer']


# In[ ]:




