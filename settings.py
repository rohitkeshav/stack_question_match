"""
Python3.5 environment
Checks if a question exists and returns the most similar question out
Base assumption: All titles are questions
Parts Implemented:
    - Get data from stacks API (currently sticking to just the title)
    - Storing in CSV
    - Clean both columns, tokenize to a usable format
"""

# Meta data
__author__ = ['Rohit Keshav', 'Anandu Anilkumar']
__license__ = 'MIT'

import os
from stackapi import StackAPI


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/stack_question_match'
SITE = StackAPI('stackoverflow')
