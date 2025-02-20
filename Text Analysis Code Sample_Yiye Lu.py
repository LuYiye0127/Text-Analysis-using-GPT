#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: luyiye
@overview: This Python script reads disclosure text data from an Excel file, filters it for specific records, and generates custom prompts based on text from different years. It then uses the OpenAI API to perform text analysis with a GPT model, processes the responses into a structured table, and saves the results as CSV and Excel files.
"""
import pandas as pd
import os
from openai import OpenAI
import logging
from ratelimit import limits, sleep_and_retry
from dotenv import load_dotenv

# Set up logging for better tracking of the execution
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Change directory to the parent directory of the script
def set_dir():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.abspath(os.path.join(current_dir, os.pardir)))

# Extract column content for a specific year from the DataFrame
def get_column_content(df, year):
    filtered_df = df[df['file_year'] == year]
    if filtered_df.empty:
        return ""
    content = filtered_df['Pay_Ratio_Paragraph'].dropna().astype(str).tolist()
    return " ".join(content)

# Generate prompts based on the input data
def write_prompts(payRatio_para_14A):
    prompt_dict = {}
    
    for cik in payRatio_para_14A['cik'].unique():
        payRatio_temp = payRatio_para_14A[payRatio_para_14A['cik'] == cik].reset_index(drop = True)
    
        para_2018 = get_column_content(payRatio_temp, 2018)
        para_2019 = get_column_content(payRatio_temp, 2019)
        para_2020 = get_column_content(payRatio_temp, 2020)
        
        prompt = f"""  

     #Context#
     The Prompt to guide GPT to analyze texts. 
     For example, You are a text analyst with professional expertise in accounting, compliance review, and regulatory disclosures. Your task is to analyze the content in #TEXT#. Please note:
     1. No Fabrication: Do not add or infer any information that is not explicitly mentioned in #TEXT#.
     2. Strict Adherence: Follow all requirements in the #Objective#, #Instructions#, #Table Column Content Requirements#, #Employee Compensation Components Category#, and #Note# sections.
     3. Output Format: Present your analysis and summary in a single table with 10 columns (no additional text). The columns must appear in the following order:
         <cik><File year><Fiscal year><compensation category><category inc_dummy><category dec_dummy><compensation element><element inc_dummy><element dec_dummy><median employee dummy>
     ......

       
     #TEXT#

         2018 description: '''{para_2018}'''
        
         2019 description: '''{para_2019}'''
        
         2020 description: '''{para_2020}'''

        """

        prompt_dict[cik] = prompt
        
    prompt_df = pd.DataFrame(list(prompt_dict.items()), columns=['cik', 'prompt'])
    prompt_df.to_csv('Outputs/prompt_df_new.csv', index = False)
    
    return prompt_dict, prompt_df

def parse_table_to_tuple(table_str):
    rows = table_str.strip().split('\n')[2:]  # Skip the first two lines (headers and dashes)
    result = []
    for row in rows:
        columns = [col.strip() for col in row.split('|')[1:-1]]  # Exclude the first and last empty columns
        if len(columns) == 10:
            result.append(tuple(columns))
    return result

@sleep_and_retry
@limits(calls=2000, period=60)
def chatFindTransDetail(apikey, prompt: str) -> str:
    try:
        client = OpenAI(api_key=apikey)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="o1-mini", #GPT model
        )
        response_data = chat_completion.choices[0].message.content
        return response_data
    except Exception as e:
        logging.error(f"Error during API call: {e}")
        return str({"error": str(e)})

#Input excel file
if __name__ == '__main__':
    set_dir()
    payRatio_para = pd.read_excel('data/Input.xlsx') # Change it based on your needs
    payRatio_para_14A = payRatio_para[payRatio_para['type'] == 'DEF 14A'].reset_index(drop=True)
    prompt_dict, prompt_df = write_prompts(payRatio_para_14A)
    gptResult_lst = []
    load_dotenv()
    apikey = os.environ.get("OPENAI_API_KEY")

    for i, (cik, prompt) in enumerate(prompt_dict.items()):
        logging.info(f"Processing CIK {cik}")
        
        gptRes = chatFindTransDetail(apikey, prompt)
        gptRes_lot = parse_table_to_tuple(gptRes)
        gptResult_lst.extend(gptRes_lot)
        if i == 1: # Number of cik to process, Change it based on the number of cik you want to process. 
            break
        print(i)

    columns = ['cik', 'File year', 'Fiscal year', 'compensation category', 'category inc_dummy', 'category dec_dummy', 'compensation element', 'element inc_dummy', 'element dec_dummy', 'median employee dummy']
    gptResult_df = pd.DataFrame(gptResult_lst, columns=columns)
    gptResult_df.to_csv('Outputs/gptResult_df.csv', index=False)
    gptResult_df.to_excel('Outputs/gptResult_df.xlsx', index=False)
    logging.info("Results saved successfully.")
