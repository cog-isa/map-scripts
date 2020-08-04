#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:32:18 2020

@author: elias
"""
from ScriptExtract.GraphScript.graph_construction import create_script_sign

def print_list(name, l, start = "\t"):
    print(name)
    for ind, i in enumerate(l):
        print(start + str(ind) + ". " + i)

name_table = 'simple_auto_table.pickle'
path = "../Texts/"
list_files = [path + 'text0_0.txt']
S, actions_sign, role_sign, obj_sign, char_sign, signifs = create_script_sign(list_files, name_table)
print_list("\nActions names:", actions_sign.keys())
print_list("\nPredicates:", role_sign.keys())
print_list("\nCharacteristics:", char_sign.keys())
print_list("\nPlaceholders:", obj_sign.keys())
print()

for key in S.significances:
    signif = S.significances[key]
    print(signif.index, signif.cause)
