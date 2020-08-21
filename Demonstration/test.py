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

name_table = 'simple_auto_table.pickle'
path = "../Texts/"
list_files = [path + 'text0_0.txt']
S, actions_sign, role_sign, obj_sign, char_sign, signifs = create_script_sign(list_files, name_table)


start, goal = get_situations(S)

class Planner:
    def __init__(self, script):
        self.script = script
        
    def get_predicates(self, signifs):
        pred_signs = []
        for event in signifs:
            coincidences = list(event.coincidences)
            for connector in coincidences:
                pred_signs.append(connector.out_sign)
        return pred_signs
    
    def try_cause(self, current_predicates, cause):
        pred_signs = self.get_predicates(cause)
        for i in pred_signs:
            if not i in current_predicates:
                return None
        
        return pred_signs
    
    def change_predicates(self, predicates, old_predicates, new_predicates):
        for pred in old_predicates:
            predicates.remove(pred)
        predicates += new_predicates
        return predicates
    
    def create_plan(self, start, goal):        
        predicates = self.get_predicates(start.significances[1].cause)
        
        plan = []
        
        for event in self.script.significances[1].cause:
            coincidences = list(event.coincidences)
            for connector in coincidences:
                act_sign = connector.out_sign
                index = connector.out_index
                act_signifs = act_sign.significances[index]
                
                cause = act_signifs.cause
                effect = act_signifs.effect
                
                pred_signs = self.try_cause(predicates, cause)
                
                if pred_signs is None:
                    print("Planning is failed")
                    return []
                
                plan.append(act_sign)
                
                new_predicates = self.get_predicates(effect)
                predicates = self.change_predicates(predicates,
                                                    pred_signs,
                                                    new_predicates)
        
        goal_pred = self.try_cause(predicates, goal.significances[1].cause)
        if not goal_pred is None:
            print("Planning is done successfully!")
            return plan
        else:
            return []
        
planner = Planner(S)
plan = planner.create_plan(start, goal)
print(plan)
#print_list("\nActions names:", actions_sign.keys())
#print_list("\nPredicates:", role_sign.keys())
#print_list("\nCharacteristics:", char_sign.keys())
#print_list("\nPlaceholders:", obj_sign.keys())
#print()
#
#print_list_rec(S)

