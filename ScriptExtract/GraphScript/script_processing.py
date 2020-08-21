#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 08:40:16 2020

@author: elias
"""

from mapcore.swm.src.components.semnet import Sign

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
            #print(connector, type(connector))
            act_sign = connector.out_sign
            #print(act_sign)
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
    goal_predicates = extract_predicates(S, effect = True)
    goal = create_situation(name = 'goal', predicates = goal_predicates)
    return start, goal