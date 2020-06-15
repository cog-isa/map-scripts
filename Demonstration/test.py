#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:32:18 2020

@author: elias
"""

from ScriptExtract.GraphScript.graph_construction import create_script_sign

name_table = 'simple_auto_table.pickle'
path = "../Texts/"
list_files = [path + 'text0_0.txt']
S, actions_sign, role_sign, obj_sign, signifs = create_script_sign(list_files, name_table)