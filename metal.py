from flask import Blueprint, render_template, request
import openai
from metal_sdk.metal import Metal
import os
import inspect
import json


metal_module = Blueprint('metal_module', __name__)
openai.api_key = "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"

api_key = "pk_jEyG+TzAS1czHqStsWJRdEghxfz+rvfpNdBzZ+mY/OU="
client_id = "ci_Klqqk9LpWdWUqA7EO2zlSYrTWLNHyRVGWXUD23Wf0pQ="
index_id = "650fbf41ded6e7b8b47766d8"

metal = Metal(api_key, client_id)

@metal_module.route('/metal', methods=['GET'])
def test_metal():
    return 'Metal'

@metal_module.route('/metal/test', methods=['GET'])
def index():
    return render_template('metal.html')

@metal_module.route('/metal/imageSearch', methods=['POST'])
def searchImages():
    prompt = request.form['prompt']
    print(f"prompt : {prompt}")
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="256x256"
    )
    image_url = response['data'][0]['url']

    results = metal.search({"imageUrl":image_url},index_id=index_id,limit=10)
    
    datas = json.loads(results.text)['data']
    print(datas) 
    
    for data in datas:
        print(data['imageUrl'])
    
    return render_template('metal.html',image_url=image_url,datas=datas)

