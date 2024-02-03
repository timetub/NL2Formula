

import xlwings

import xlwings as xw

import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

import json

# read a json file
def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)



from thefuzz import fuzz
from thefuzz import process

def fuzzy_cmp_str(a:str,b:str):
    a,b=str(a).lower(),str(b).lower()
    return fuzz.partial_ratio(a, b)

def fuzzy_cmp_str(a:str,b:str):
    a,b=str(a).lower(),str(b).lower()
    return fuzz.partial_ratio(a, b)

def get_formular(question: str,table, model='gpt-3.5-turbo'):
    llm = ChatOpenAI(model_name=model, max_tokens=50,)
    prompt = "Write a excel formula to answer the quesion based on table beleow\n"\
        "You can use functions like SUM,XLOOKUP,UNIQUE,AVERAGE,etc...\n"\
             "Question:{question}\n" \
             "Here is the Table: {table}.\n" \
             "{format_instructions}"
             
    prompt_template = ChatPromptTemplate.from_template(prompt)
    
    #options = ResponseSchema(name="words", description="The words are related to bad behavior.")
    options = ResponseSchema(name="Formula", description="The formula is related to question")
    response_schemas = [options]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    form_instructions = output_parser.get_format_instructions()
    
    final_prompt = prompt_template.format_messages(question=question, table=table, format_instructions=form_instructions)
    print(final_prompt)
    output = llm(final_prompt)
    return output_parser.parse(output.content)






def save_table(sheet,table,):
    processed_data = [row[1:] for row in table[1:]]
    table_data= [[element.lower() for element in row] for row in processed_data]
    sheet.clear()
    num_rows = len(table_data)
    num_columns = len(table_data[0])

    fill_range = sheet.range((1, 1), (num_rows+5, num_columns+5))
    fill_range.number_format = '@'
    fill_range.value=table_data 
    
    
    
def run_execution_test(test_list,sheet):
    count,count_true=0,0
    em,exe=0,0
    count_wronggd=0
    fail_list=[]
    for item in test_list[:]:
        pre_f = item.get('pre_f')
        if pre_f is None:
            item['res']='not found'
            continue
        count+=1
        formula2 = item['Formula2']
        a=formula2.lower().replace(" ", "").replace("/", "")
        b=pre_f.lower().replace(" ", "").replace("/", "")
        nl=item['Question']
        if fuzzy_cmp_str(formula2,pre_f)>95:
            item['res']='exact_match'
            count_true_em+=1
            count_true+=1
            em+=1
            print(formula2,"\n",pre_f)
            print(count,'exact_match')
        flag,gdres,llmres=eval_execution(sheet,item,item['Table'],formula2,pre_f)
        item['gdres'],item['gpt3res']=gdres,llmres
        if(flag):
            item['res']='exe_match'
            count_true+=1
            count_true_exe+=1
            # elif (result=='wrong_groundtruth'):
            #     count_wronggd+=1
            #     print(count_wronggd)
        else:
            item['res']='failed'
            fail_list.append(item)
            #print(f"failed",f"{nl}\n{a}\n{b}")    
            #print(count_true/count)
    print(em)
    print(exe)
    return fail_list,count_true/count,count_wronggd
    


def run_gpt_execution_test(test_list,sheet):
    count,count_true=0,0
    em,exe=0,0
    count_wronggd=0
    fail_list=[]
    for item in test_list[:]:
        pre_f = item.get('pre_f')
        if pre_f is None:
            item['res']='not found'
            continue
        count+=1
        formula2 = item['Formula2']
        #llmformula=item['llmFormula']
        a=formula2.lower().replace(" ", "").replace("/", "")
        b=pre_f.lower().replace(" ", "").replace("/", "")
        #c=llmformula.lower().replace(" ", "").replace("/", "")
        nl=item['Question']
        if fuzzy_cmp_str(formula2,pre_f)>95:
            item['res']='exact_match'
            count_true+=1
            em+=1
            #print(count,'exact_match')
        flag,gdres,llmres=eval_execution(sheet,item,item['Table'],formula2,pre_f)
        item['gdres'],item['gpt3res']=gdres,llmres
        if(flag):
            item['res']='exe_match'
            count_true+=1
            # elif (result=='wrong_groundtruth'):
            #     count_wronggd+=1
            #     print(count_wronggd)
        else:
            item['res']='failed'
            fail_list.append(item)
            #print(f"failed",f"{nl}\n{a}\n{b}")    
            #print(count_true/count)
    print(em)
    print(exe)
    return fail_list,count_true/count,count_wronggd




    