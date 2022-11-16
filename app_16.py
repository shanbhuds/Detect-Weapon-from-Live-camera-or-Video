# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 20:02:30 2022

@author: 103077
"""

from flask import Flask,jsonify,request

app = Flask(__name__)

@app.route('/')
def detect():
    return "<H1>Hello world</H1>"

app.run()
    

