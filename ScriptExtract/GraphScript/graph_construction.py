#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 06:06:57 2020

@author: elias
"""

import re

import numpy as np
import pymorphy2

from ScriptExtract.Preprocessing.TextProcessing import Table
from ScriptExtract.SiteParser.site_parser import Parser
from mapcore.swm.src.components.semnet import Sign

def get_feature_dict(table, key_word = "depend_lemma"):
    full_list_actions = []
    sentences = []
    relations = []
    for key in table:
        table_text = table[key]
        help_list = []
        help_sentences = []
        help_relations = []
        for item in table_text:
            for act in item['Actions']:
                help_sentences.append(item["Sentence"])
                help_relations.append(item['Relations'])
                help_list.append(act)
        sentences.append(help_sentences)
        relations.append(help_relations)
        full_list_actions.append(help_list)
    verb_dict = dict()
    feature_dict = dict()
    N = 0
    N_verb = 0
    for ind, l in enumerate(full_list_actions):
        for ind1, act in enumerate(l):
            sentence = sentences[ind][ind1]
            if act.inform['VERB'] is not None:
                act.inform['SEM_REL'] = relations[ind][ind1]
                b, e = act.inform['VERB'][0].begin, act.inform['VERB'][0].end
                N_verb += 1
                verb = sentence[b:e]
                if verb in verb_dict:
                    verb_dict[verb][(ind,ind1)] = 0
                else:
                    verb_dict[verb] = {(ind,ind1):0}
                for depend in act.inform:
                    if not depend in ['punct', 'VERB', 'SEM_REL']:
                        for w in act.inform[depend]:
                            N += 1
                            lemma = w[0].lemma
                            if not w[0].postag in ['CONJ', 'PRON', 'VERB']:
                                if key_word == "lemma":
                                    if lemma in feature_dict:
                                        feature_dict[lemma][(ind,ind1)] = 0
                                    else:
                                        feature_dict[lemma] = {(ind,ind1):0}
                                if key_word == "depend_lemma":
                                    if (depend, lemma) in feature_dict:
                                        feature_dict[(depend, lemma)][(ind,ind1)] = 0
                                    else:
                                        feature_dict[(depend, lemma)] = {(ind,ind1):0}     
                if key_word == 'sem_rel':
                    rel = relations[ind][ind1]
                    for r in rel:
                        parent, child = r['parent'], r['child']
                        if (act.inform['VERB'][0].begin == parent['start'] and
                            act.inform['VERB'][0].end == parent['end']):
                            for j in act.inform:
                                if j != 'VERB' and j != 'SEM_REL':
                                    for word in act.inform[j]:
                                        if (word[0].begin == child['start'] and
                                            word[0].end == child['end']):
                                            lemma = word[0].lemma
                                            tp = r['tp']
                                            if lemma in feature_dict:
                                                feature_dict[lemma][(ind,ind1)] = tp
                                            else:
                                                feature_dict[lemma] = {(ind,ind1):tp}
    return full_list_actions, verb_dict, feature_dict

def create_table_of_sets(feature_dict, full_list_actions):
    list_of_set = []
    for i in feature_dict:
        ind_actions = feature_dict[i].keys()
        set_cup = []
        for ind in ind_actions:
            for ind_s, s in enumerate(list_of_set):
                if ind in s:
                    set_cup.append(ind_s)
        if len(set_cup) == 0:
            list_of_set.append(set(ind_actions))
        else:
            new_set = set([i for j in set_cup for i in list_of_set[j]])
            list_of_set = [i for ind, i in enumerate(list_of_set) if not ind in set_cup]
            list_of_set.append(new_set)
    return [[(ind,ind1) for ind,ind1 in j] for j in list_of_set]

def _add_start(E, V, start, docs):
    V.append(start)
    E[start] = []
    start_docs = [min([i[1] for i in V if i[0] == j]) for j in docs]
    for ind, i in enumerate(start_docs):
        v = docs[ind], i
        E[start].append(v)
            
def _add_end(E, V, end, docs):
    V.append(end)
    E[end] = []
    end_docs = [max([i[1] for i in V if i[0] == j]) for j in docs]
    for ind, i in enumerate(end_docs):
        v = docs[ind], i
        E[v].append(end)

def construct_graph(table,
                    key_word = "depend_lemma",
                    with_next = False,
                    start = (-1,-1), end = (-2,-2), 
                    min_set = 2, max_set = np.infty
                    ):
    """
    'table' is The result of 
        ScriptExtract.Preprocessing.TextProcessing.Table().get_table
    'key_word' is type of analysed verb argument
    'with_next' is booolean. If True Construct edges 
        between neibourghood in one document.
    'start' and 'end' are tuples of two integers. It is start and end vertices in graph.
    """
    
    full_list_actions, verb_dict, feature_dict = get_feature_dict(table, key_word)
    table = create_table_of_sets(feature_dict, full_list_actions)
    table = [i for i in table if max_set>len(i) >= min_set]
    
    # Vertices list
    V = [i for j in table for i in j]
    V.sort(key = lambda x: x[0])
    
    # Dictionary for edges (adjacency list)
    E = {v:[] for v in V}
    
    # Edges between neibourghood in one document
    if with_next:
        for ind, v in enumerate(V):
            if ind+1 < len(V) and V[ind+1][0] > v[0]:
                E[v].append(V[ind+1])
    
    # Edges between action with same arguments
    for i in table:
        for v in i:
            for v_ in i:
                if v_ != v and not v_ in E[v]:
                    E[v].append(v_) 
                    
    # Adding start and end vertices
    docs = np.unique(np.array([i[0] for i in V]))
    if start is None:
        start = (-1, -1)
    if end is None:
        end = (-2, -2)
    _add_start(E, V, start, docs)
    _add_end(E, V, end, docs)
    return (E, V), (start, end), (full_list_actions, verb_dict, feature_dict)

def graph_inform(V, E):
    s = 0
    s1 = 0
    for i in E:
        s+=len(E[i])
        s1 += len([j for j in E[i] if not(j[0] == i[0] and j[1] == i[1]+1)])
    print("The edges number", s)
    print("The out-edges number", s1)
    print("The vertices number", len(V))
    print("The number of document", len(np.unique(np.array([i[0] for i in V]))))
	
	

def equal_word(word, word1):
    return word.begin == word1['start'] and word.end == word1['end']

class Script:
    def __init__(self, V, E, full_list_actions, start = (-1,-1), end = (-2,-2)):
        self.V, self.E = V, E
        self.start = start
        self.end = end
        self.V_inform, self.E_inform = self._reconstruct(V, E, full_list_actions)
        self.V_descr = {}
        self.bad_BFS()

    def _reconstruct(self, V, E, full_list_actions):
        E = E
        V = {v: self._get_inform(v, full_list_actions) for v in V}
        return V, E
    
    def bad_BFS(self):
        start, end = self.start, self.end
        self.V.sort(key = lambda x: x[0])
        self.V = [i for i in self.V if i!= start and i!=end]
        self.V = [start] + self.V + [end]
        self.V_descr[start] = {}
        #previous = start
        for v in self.V[1:]:
            if v in self.E and len(self.E[v]) >0 or v == end:
                #self.V_descr[v] = copy.deepcopy(self.V_descr[previous])
                self.V_descr[v] = {}
                self.update(v)
                #previous = v
        return None
    
    def update(self, v):
        inform = self.V_inform[v]
        if 'Sentence' in inform:
            sentence = inform['Sentence']
            self.V_descr[v]['Sentence'] = sentence
        else:
            self.V_descr[v]['Sentence'] = ''
        if 'VERB' in inform:
            self.V_descr[v]['verb'] = inform['VERB']
        if 'локатив' in inform:
            self.V_descr[v]['локатив'] = {'локатив':inform['локатив'],
                                          'Sentence':sentence,
                                          'verb': inform['VERB']}
        if 'объект' in inform:
            obj = {'объект':inform['объект'],
                    'verb':inform['VERB'],
                  'Sentence':sentence}
            if 'объект' in self.V_descr[v]:
                self.V_descr[v]['объект'].append(obj)
            else:
                self.V_descr[v]['объект'] = [obj]
        if 'субъект' in inform:
            subj = {'субъект':inform['субъект'],
                    'verb':inform['VERB'],
                   'Sentence':sentence}
            if 'субъект' in self.V_descr[v]:
                self.V_descr[v]['субъект'].append(subj)
            else:
                self.V_descr[v]['субъект'] = [subj]
        if 'темпоратив' in inform:
            self.V_descr[v]['темпоратив'] = {'темпоратив':inform['темпоратив'],
                                          'Sentence':sentence,
                                          'verb': inform['VERB']}
    
    def extract_predicates(self, index, synt_tree):
        list_of_vert = synt_tree.kids
        root = None
        while len(list_of_vert) > 0:
            ch, tp = list_of_vert.pop(0)
            if not ch is None and ch.value.index == index:
                root = ch
                break
            list_of_vert += ch.kids
        if root is None:
            return []
        predicates = []
        tps = ["amod", "appos", "nmod"]
        for ch, tp in root.kids:
            if tp in tps:
                predicat = ch.value.lemma + " (" + root.value.lemma + ")" + "[" + tp + "]"
                predicates.append(predicat)
        return predicates
        
    def _get_inform(self, v, full_list_actions):
        if v == self.start or v == self.end:
            return {}
        act = full_list_actions[v[0]][v[1]]
        inform = {}
        inform['Sentence'] = act.sentence
        inform['VERB'] = act.inform['VERB']
        for j in act.inform['SEM_REL']:
            type_rel = j['tp']
            parent = j['parent']
            child = j['child']
            if equal_word(act.inform['VERB'][0], parent):
                for key in act.inform:
                    if not key in ['VERB', 'SEM_REL']:
                        for word, list_depend, _ in act.inform[key]:
                            if equal_word(word, child):
                                if type_rel == "объект":
                                    synt_tree = act.synt_tree
                                    predicates = self.extract_predicates(word.index, synt_tree)
                                else:
                                    predicates = []
                                if type_rel in inform:
                                    inform[type_rel].append((word, list_depend, predicates))
                                else:
                                    inform[type_rel] = [(word, list_depend, predicates)]
        return inform
    
    def GetNext(self, v):
        return self.E[v]
	
def get_srcipt(list_files, name_table = "1.pickle", key_word = "sem_rel"):
	table_ = Table().get_table(list_files, test = lambda act: True, name_table = name_table)
	(E, V), (start, end), (full_list_actions, verb_dict, feature_dict) = construct_graph(table_, key_word = key_word)
	script_ = Script(V,E, full_list_actions)
	return script_

def union(sentence, list_):
    q = list(np.array(sentence)[list_])
    return ' '.join(q)

def print_dict(v_desr):
    if 'Sentence' in v_desr:
        print(' '.join(v_desr['Sentence']))
    if 'локатив' in v_desr:
        print('\nлокатив:'.upper())
        print("%s (%s)"%(v_desr['локатив']['локатив'][0][0].lemma,
                         union(v_desr['локатив']['Sentence'], v_desr['локатив']['локатив'][0][1])))
    if 'темпоратив' in v_desr:
        print('\nтемпоратив:'.upper())
        print("%s (%s)"%(v_desr['темпоратив'][0].lemma, union(v_desr['Sentence'], v_desr['темпоратив'][1])))
    if 'объект' in v_desr:
        print('\nобъект'.upper())
        obj = v_desr['объект']
        for i in obj:
            for j in i['объект']:
                print("%s (%s) - %s (%s)"%(j[0].lemma, union(i['Sentence'], j[1]),
                                           i['verb'][0].lemma, union(i['Sentence'], i['verb'][1])))
    if 'субъект' in v_desr:
        print('\nсубъект'.upper())
        obj = v_desr['субъект']
        for i in obj:
            for j in i['субъект']:
                print("%s (%s) - %s (%s)"%(j[0].lemma, union(i['Sentence'], j[1]),
                                           i['verb'][0].lemma, union(i['Sentence'], i['verb'][1])))

def parsing_predicate(pred_name):
    full_name_obj = re.search(r"\(.*\)", pred_name).group(0)[1:-1]
    char_name = re.search(r".*\(", pred_name).group(0)[:-2]
    return char_name, full_name_obj

def _add_signifs(name_act, full_name_obj, role_name,
                actions_sign = {}, role_sign = {}, obj_sign = {}, signifs = {}, char_sign = {},
                is_predicate = False):
    """
    The function has to add events into cause of action 'name_act'
    
    The input of function:
        name_act (string)
            Name of action. Invinitive of corresponding verb
        full_name_obj (string)
            Name of object.
        role_name (string)
            The name of role
        S (Sign)
            the Sign of Script
        actions_sign (dict)
            the dictionary with values are signs of actions and keys are their
            names
        obj_sign (dict)
            the dictionary with values are signs of placeholders and keys are their
            names        
        char_sign (dict)
            the dictionary with values are signs of characteristics and keys are their
            names        
        role_sign (dict)
            the dictionary with values are roles of characteristics and keys are their
            names        
        signifs (dict)
            the dictionary with values are causal matrices of significances and keys are their
            names
    """
    if (not role_name in role_sign) and (not role_name is None):
        role_sign[role_name] = Sign(role_name)
        signifs[role_name] = role_sign[role_name].add_significance()
    elif role_name is None:
        # create connector for temporativ and locativ
        if not full_name_obj in role_sign:
            role_sign[full_name_obj] = Sign(full_name_obj)
            signifs[full_name_obj] = role_sign[full_name_obj].add_significance()
            new_predicate = True
        else:
            new_predicate = False
        connector = signifs[name_act].add_feature(signifs[full_name_obj], zero_out=True)
        role_sign[full_name_obj].add_out_significance(connector)
        if is_predicate and new_predicate:
            predicate_name = full_name_obj
            char_name, full_name_obj = parsing_predicate(predicate_name)
            if not full_name_obj in obj_sign:
                obj_sign[full_name_obj] = Sign(full_name_obj)
                signifs[full_name_obj] = obj_sign[full_name_obj].add_significance()
            connector = signifs[predicate_name].add_feature(signifs[full_name_obj], zero_out=True)
            obj_sign[full_name_obj].add_out_significance(connector)
            if not char_name in char_sign:
                char_sign[char_name] = Sign(char_name)
                signifs[char_name] = char_sign[char_name].add_significance()
            connector = signifs[predicate_name].add_feature(signifs[char_name], zero_out=True)
            char_sign[char_name].add_out_significance(connector)
        return
    
    # action -> locativ
    connector = signifs[name_act].add_feature(signifs[role_name], zero_out=True)
    role_sign[role_name].add_out_significance(connector)
    
    # locativ -> placeholder
    if not full_name_obj in obj_sign:
        obj_sign[full_name_obj] = Sign(full_name_obj)
        signifs[full_name_obj] = obj_sign[full_name_obj].add_significance()
    signifs[role_name] = role_sign[role_name].add_significance()
    connector = signifs[role_name].add_feature(signifs[full_name_obj], zero_out=True)
    obj_sign[full_name_obj].add_out_significance(connector)

def _add_signifs_effect_action(name_act, full_name_obj, add_name_act,
                actions_sign = {}, role_sign = {}, obj_sign = {}, char_sign = {}, signifs = {}):
    """
    The function has to add events into effect of action 'name_act'
    
    The input of function:
        name_act (string)
            Name of action. Invinitive of corresponding verb
        full_name_obj (string)
            Name of object.
        add_name_act (string)
            The form of action from text. It is used for constructing participle
        S (Sign)
            the Sign of Script
        actions_sign (dict)
            the dictionary with values are signs of actions and keys are their
            names
        obj_sign (dict)
            the dictionary with values are signs of placeholders and keys are their
            names        
        char_sign (dict)
            the dictionary with values are signs of characteristics and keys are their
            names        
        role_sign (dict)
            the dictionary with values are roles of characteristics and keys are their
            names        
        signifs (dict)
            the dictionary with values are causal matrices of significances and keys are their
            names
    """
    try:
        morph = pymorphy2.MorphAnalyzer()
        char_name = morph.parse(add_name_act)[0].inflect({'PRTF', 'perf', 'pssv', 'past'}).word
    except Exception:
        char_name = name_act + "_PRTF_pssv_past_perf"
        
    pred_name = "{not}" + char_name + "(" + full_name_obj + ")" + "[amod]"
    if not pred_name in role_sign:
        role_sign[pred_name] = Sign(pred_name)
    pred_sign = role_sign[pred_name]

    signifs[pred_name] = pred_sign.add_significance()
    
    # action -> predicat
    connector = signifs[name_act].add_feature(signifs[pred_name], zero_out=True, effect = False)
    pred_sign.add_out_significance(connector)

    if not full_name_obj in obj_sign:
        obj_sign[full_name_obj] = Sign(full_name_obj)
        signifs[full_name_obj] = obj_sign[full_name_obj].add_significance()
    connector = signifs[pred_name].add_feature(signifs[full_name_obj], zero_out=True)
    obj_sign[full_name_obj].add_out_significance(connector)
    
    try:
        morph = pymorphy2.MorphAnalyzer()
        char_name = morph.parse(add_name_act)[0].inflect({'PRTF', 'perf', 'pssv', 'past'}).word
    except Exception:
        char_name = name_act + "_PRTF_pssv_past_perf"
    if not char_name in char_sign:
        char_sign[char_name] = Sign(char_name)
        signifs[char_name] = char_sign[char_name].add_significance()
    connector = signifs[pred_name].add_feature(signifs[char_name], zero_out=True)
    char_sign[char_name].add_out_significance(connector)

    pred_name = char_name + "(" + full_name_obj + ")" + "[amod]"
    if not pred_name in role_sign:
        role_sign[pred_name] = Sign(pred_name)
    pred_sign = role_sign[pred_name]

    signifs[pred_name] = pred_sign.add_significance()
    
    # action -> predicat
    connector = signifs[name_act].add_feature(signifs[pred_name], zero_out=True, effect = True)
    pred_sign.add_out_significance(connector)

    if not full_name_obj in obj_sign:
        obj_sign[full_name_obj] = Sign(full_name_obj)
        signifs[full_name_obj] = obj_sign[full_name_obj].add_significance()
    connector = signifs[pred_name].add_feature(signifs[full_name_obj], zero_out=True)
    obj_sign[full_name_obj].add_out_significance(connector)
    
    try:
        morph = pymorphy2.MorphAnalyzer()
        char_name = morph.parse(add_name_act)[0].inflect({'PRTF', 'perf', 'pssv', 'past'}).word
    except Exception:
        char_name = name_act + "_PRTF_pssv_past_perf"
    if not char_name in char_sign:
        char_sign[char_name] = Sign(char_name)
        signifs[char_name] = char_sign[char_name].add_significance()
    connector = signifs[pred_name].add_feature(signifs[char_name], zero_out=True)
    char_sign[char_name].add_out_significance(connector)

def add_signifs(v_descr,
                S = None,
                actions_sign = {}, role_sign = {}, obj_sign = {}, char_sign = {},
                signifs = {},
                script_name = "Script",
                locativ_name = None, temporativ_name = None,
                subj_name = None, obj_name= 'объект',
				order = None):
    if 'Sentence' in v_descr:
        sentence = v_descr['Sentence']
    else:
        return
    
    if 'субъект' in v_descr:
        obj = v_descr['субъект']
        stop = True
        for i in obj:
            for j in i['субъект']:
                lemma_subj = j[0].lemma
                if lemma_subj.lower() == "я":
                    stop = False
        if stop:
            return
    
    if 'verb' in v_descr:
        name_act = v_descr['verb'][0].lemma
        add_name_act = sentence[v_descr['verb'][0].index]
        if not name_act in actions_sign:
            actions_sign[name_act] = Sign(name_act)
            signifs[name_act] = actions_sign[name_act].add_significance()
        try:
            connector_script = signifs[script_name].add_feature(signifs[name_act],
                                                       order = order,
                                                       zero_out=True)
        except Exception:
            connector_script = signifs[script_name].add_feature(signifs[name_act],
                                                       order = None,
                                                       zero_out=True)
        actions_sign[name_act].add_out_significance(connector_script)
    else:
        return
    
    signifs[name_act] = actions_sign[name_act].add_significance()
    
    if 'локатив' in v_descr:
        full_name_obj = v_descr['локатив']['локатив'][0][0].lemma
        full_locativ = union(sentence, v_descr['локатив']['локатив'][0][1])
        _add_signifs(name_act, full_locativ, locativ_name,
                     actions_sign = actions_sign,
                     role_sign = role_sign,
                     obj_sign = obj_sign, char_sign = char_sign,
                     signifs = signifs)
        predicate_name = full_locativ
        if not full_name_obj in obj_sign:
            obj_sign[full_name_obj] = Sign(full_name_obj)
            signifs[full_name_obj] = obj_sign[full_name_obj].add_significance()
        connector = signifs[predicate_name].add_feature(signifs[full_name_obj], zero_out=True)
        obj_sign[full_name_obj].add_out_significance(connector)
        
    if 'темпоратив' in v_descr:
        full_name_obj = v_descr['темпоратив'][1][0].lemma
        full_temporativ = union(sentence, v_descr['темпоратив'][1])
        _add_signifs(name_act, full_temporativ, temporativ_name,
                     actions_sign = actions_sign,
                     role_sign = role_sign,
                     obj_sign = obj_sign, char_sign = char_sign,
                     signifs = signifs)
        predicate_name = full_temporativ
        if not full_name_obj in obj_sign:
            obj_sign[full_name_obj] = Sign(full_name_obj)
            signifs[full_name_obj] = obj_sign[full_name_obj].add_significance()
        connector = signifs[predicate_name].add_feature(signifs[full_name_obj], zero_out=True)
        obj_sign[full_name_obj].add_out_significance(connector)
        
    if 'объект' in v_descr:
        obj = v_descr['объект']
        for i in obj:
            for j in i['объект']:
                lemma_obj = j[0].lemma
                full_obj = union(sentence, j[1])
                #root = get_tree(full_obj)[0]
                #print("\n"+full_obj, get_tree(full_obj))
                #print(root.value.lemma)
                #for child, type_ in root.kids:
                #    print('-', child.value.lemma, type_)
                predicates = j[2]
                for predicate in  predicates:
                    _add_signifs(name_act, predicate, None,
                                 actions_sign = actions_sign,
                                 role_sign = role_sign,
                                 obj_sign = obj_sign, char_sign = char_sign,
                                 signifs = signifs, is_predicate = True)
#                _add_signifs(name_act, lemma_obj, obj_name,
#                             actions_sign = actions_sign,
#                             role_sign = role_sign,
#                             obj_sign = obj_sign,
#                             signifs = signifs)
                _add_signifs_effect_action(name_act, lemma_obj,
                                           add_name_act,
                             actions_sign = actions_sign,
                             role_sign = role_sign,
                             obj_sign = obj_sign,
                             char_sign = char_sign,
                             signifs = signifs)
    """
    if 'субъект' in v_descr:
        obj = v_descr['субъект']
        for i in obj:
            for j in i['субъект']:
                lemma_subj = j[0].lemma
                full_subj = union(sentence, j[1])
                _add_signifs(name_act, lemma_subj, subj_name,
                             actions_sign = actions_sign,
                             role_sign = role_sign,
                             obj_sign = obj_sign, char_sign = char_sign,
                             signifs = signifs)
    """
    return connector_script

def add_obj_link(obj_name, word, link = "hyper", obj_sign = {}, signifs = {}):
    if not word in obj_sign:
        obj_sign[word] = Sign(word)
        signifs[word] = obj_sign[word].add_significance()
        
    if link == "hyper" or link == "syno":
        connector = signifs[word].add_feature(signifs[obj_name], zero_out=True)
        obj_sign[obj_name].add_out_significance(connector)
        
    if link == "hypo" or link == "syno":
        connector = signifs[obj_name].add_feature(signifs[word], zero_out=True)
        obj_sign[word].add_out_significance(connector)

def create_script_sign(list_files, name_table= None, key_word = "sem_rel", script_name = "Script"):
    """
    This function creates Script and the required signs for it and 
    for the actions, roles and their possible placeholders (objects).
    
    The adding of connectors in Script is implemented in function add_significance
    
    This function returns the Sign of Script (S),
                        the signs of actions (dict 'actions_sign'),
                        the role signs (dict 'role_sign'),
                        the placeholders signs (dict 'obj_sign'),
                        characteristics signs (dict 'char_sign')
                        significances (dict 'signifs')
    """
    
    if name_table is None:
        name_table = "DELETE.pickle"
    	
    # Extract semantic relations and syntactic dependences for all texts in list_files
    # into table
    graph_script = get_srcipt(list_files, name_table)
    V_descr = graph_script.V_descr
    
    # The script sign
    S = Sign(script_name)
    
    # The keys of the following dictionaries are signs name. The value is sign
    actions_sign = {}
    role_sign = {}
    obj_sign = {}
    char_sign = {}
    signifs = {}
    
    signifs[script_name] = S.add_significance()
    order = None
    for v in graph_script.V:
        connector_script = add_signifs(V_descr[v],
                    S = S,
                    signifs = signifs,
                    script_name = script_name,
                    actions_sign = actions_sign,
                    role_sign = role_sign,
                    obj_sign = obj_sign,
                    char_sign = char_sign,
					order = order)
    obj_sign_names = list(obj_sign.keys())
    parser = Parser()
    hyperonyms_key = "Гиперонимы"
    hyponyms_key = "Гипонимы"
    synonyms_key = "Синонимы"
    for obj_name in obj_sign_names:
        response = parser.get_word_info(obj_name)
        hyperonyms = response[hyperonyms_key]

        if hyperonyms is None:
            hyperonyms = []
        for word in hyperonyms:
            add_obj_link(obj_name, word, link = "hyper", obj_sign = obj_sign, signifs = signifs)
        hyponyms = response[hyponyms_key]

        if hyponyms is None:
            hyponyms = []
        for word in hyponyms:
            add_obj_link(obj_name, word, link = "hypo", obj_sign = obj_sign, signifs = signifs)
        #synonyms = response[synonyms_key]
        #if synonyms is None:
        #    synonyms = []
        #for word in synonyms:
        #    add_obj_link(obj_name, word, link = "syno", obj_sign = obj_sign, signifs = signifs)
    return S, actions_sign, role_sign, obj_sign, char_sign, signifs