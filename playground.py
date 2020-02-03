import os
import json

def get_file_index(filesProcessed):
    new_dict = {}
    for f in filesProcessed:
        new_dict[f]={"framerate": 30.0, "selected": {"0": 1, "9000": 0}}
    return new_dict

output_file = json.load(open("/data/yijunq/results/output.json","r"))

file_dict = get_file_index(output_file["filesProcessed"])
json_str = json.dumps(file_dict,indent=4)
with open(os.path.join("/data/yijunq/results","file-index.json"), 'w') as save_json:
    save_json.write(json_str)