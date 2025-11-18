#!/usr/bin/python
# -*- coding: UTF-8 -*-


from openai import OpenAI

M_QWEN_8B = "qwen/qwen3-8b"

def llm_msgs_builder(sys_prompt = None, user_prompt = None): 
    msgs = []
    if sys_prompt != None: 
        msgs.append({"role": "system", "content": sys_prompt})
    
    if user_prompt != None: 
        msgs.append({"role": "user", "content": sys_prompt})

    return msgs


