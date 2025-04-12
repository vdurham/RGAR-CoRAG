import json
import os
import re
import json
import torch
from RGAR import RGAR
class QADataset:

    def __init__(self, data, dir="."):
        self.data = data.lower().split("_")[0]
        benchmark = json.load(open(os.path.join(dir, "benchmark.json")))
        if self.data not in benchmark:
            raise KeyError("{:s} not supported".format(data))
        self.dataset = benchmark[self.data]
        self.index = sorted(self.dataset.keys())

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.dataset[self.index[key]]
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")
        

dataset_name = "medqa"
dataset = QADataset(dataset_name,dir="MIRAGE")

debug_idx = 3
data = dataset[debug_idx]
question = data["question"]
options = data["options"]
correct_answer = data["answer"]

# 打印调试信息
print(f"Debugging Question {debug_idx + 1}:")
print(f"Question: {question}")
print(f"Options: {options}")
print(f"Correct Answer: {correct_answer}")

torch.cuda.empty_cache()
rgar = RGAR(
        llm_name="meta-llama/Llama-3.2-3B-Instruct", 
        retriever_name="MedCPT", 
        corpus_name="Textbooks", 
        device="cuda:0",
        cot=False,
        rag=True,
        me=2
    )
rgar.answer(question, options)