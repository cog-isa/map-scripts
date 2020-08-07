#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:32:18 2020

@author: elias
"""
from ScriptExtract.GraphScript.graph_construction import create_script_sign
from mapcore.swm.src.components.semnet import Sign
from mapcore.swm.src.components.semnet import Event

def print_list(name, l, start = "\t"):
    print(name)
    for ind, i in enumerate(l):
        print(start + str(ind) + ". " + i)
        
def print_list_rec(sign, n=0):
    print(n*"\t" + sign.name)
    for key in sign.significances:
        signif = sign.significances[key]
        print()
        print(n*"\t" + "-Cause")
        for i in signif.cause:
            print_list_rec(list(i.coincidences)[0].out_sign, n+1)
            
        print()
        print(n*"\t" + "-Effect")
        for i in signif.effect:
            print_list_rec(list(i.coincidences)[0].out_sign, n+1)
        print()
        
def _extract_predicates(act_sign, out_index, effect = False):
    if not effect:
        signifs = act_sign.significances[out_index].cause
    else:
        signifs = act_sign.significances[out_index].effect
    pred_signs = []
    for event in signifs:
        coincidences = list(event.coincidences)
        for connector in coincidences:
            pred_signs.append(connector.out_sign)
    return pred_signs

def extract_predicates(S, effect = False):
    pred_signs = []
    for event in S.significances[1].cause:
        coincidences = list(event.coincidences)
        for connector in coincidences:
            print(connector, type(connector))
            act_sign = connector.out_sign
            print(act_sign)
            pred_signs += _extract_predicates(act_sign, connector.out_index, effect = effect)
    return pred_signs

def create_situation(name, predicates):
    sign = Sign(name)
    signif = sign.add_significance()
    for pred in predicates:
        connector = signif.add_feature(pred.significances[1])
        pred.add_out_significance(connector)
    return sign

def get_situations(S):
    start_predicates = extract_predicates(S, effect = False)
    start = create_situation(name = 'start', predicates = start_predicates)
    end_predicates = extract_predicates(S, effect = False)
    end = create_situation(name = 'start', predicates = end_predicates)
    return start, end

name_table = 'simple_auto_table.pickle'
path = "../Texts/"
list_files = [path + 'text0_0.txt']
S, actions_sign, role_sign, obj_sign, char_sign, signifs = create_script_sign(list_files, name_table)

start, end = get_situations(S)



#print_list("\nActions names:", actions_sign.keys())
#print_list("\nPredicates:", role_sign.keys())
#print_list("\nCharacteristics:", char_sign.keys())
#print_list("\nPlaceholders:", obj_sign.keys())
#print()
#
#print_list_rec(S)

