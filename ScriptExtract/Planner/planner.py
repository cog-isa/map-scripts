#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 08:38:57 2020

@author: elias
"""

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