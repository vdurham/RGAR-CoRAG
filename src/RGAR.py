# This file is adapted from [Teddy-XiongGZ/MedRAG]
# Original source: [https://github.com/Teddy-XiongGZ/MedRAG/blob/main/src/medrag.py]
# we developed RGAR based on MedRAG
# we developed RAG systems without CoT based on MedRAG
# we add support for qwens
import sys
sys.path.append("src")

from config import config
from template import *
from utils import RetrievalSystem, DocExtracter
import os
import re
import json
import torch
import transformers
from transformers import AutoTokenizer
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
# imports for added CoRAG methods
from copy import deepcopy
from typing import Optional, List, Dict, Tuple
from data_utils import format_input_context, parse_answer_logprobs
from prompts import get_generate_subquery_prompt, get_generate_intermediate_answer_prompt, get_generate_final_answer_prompt
from agent_utils import RagPath
from utils import batch_truncate


def _normalize_subquery(subquery: str) -> str:
    subquery = subquery.strip()
    if subquery.startswith('"') and subquery.endswith('"'):
        subquery = subquery[1:-1]
    if subquery.startswith('Intermediate query'):
        subquery = re.sub(r'^Intermediate query \d+: ', '', subquery)

    return subquery

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Model generation timeout!")


openai.api_type = openai.api_type or os.getenv(
    "OPENAI_API_TYPE") or config.get("api_type")
openai.api_version = openai.api_version or os.getenv(
    "OPENAI_API_VERSION") or config.get("api_version")
openai.api_key = openai.api_key or os.getenv(
    'OPENAI_API_KEY') or config["api_key"]

if openai.__version__.startswith("0"):
    openai.api_base = openai.api_base or os.getenv(
        "OPENAI_API_BASE") or config.get("api_base")
    if openai.api_type == "azure":
        openai_client = lambda **x: openai.ChatCompletion.create(
            **{'engine' if k == 'model' else k: v for k, v in x.items()})["choices"][0]["message"]["content"]
    else:
        openai_client = lambda **x: openai.ChatCompletion.create(
            **x)["choices"][0]["message"]["content"]
else:
    if openai.api_type == "azure":
        openai.azure_endpoint = openai.azure_endpoint or os.getenv(
            "OPENAI_ENDPOINT") or config.get("azure_endpoint")
        openai_client = lambda **x: openai.AzureOpenAI(
            api_version=openai.api_version,
            azure_endpoint=openai.azure_endpoint,
            api_key=openai.api_key,
        ).chat.completions.create(**x).choices[0].message.content
    else:
        openai_client = lambda **x: openai.OpenAI(
            api_key=openai.api_key,
        ).chat.completions.create(**x).choices[0].message.content


class RGAR:

    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", rag=True, follow_up=False, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", cache_dir=None, corpus_cache=False, HNSW=False, device="auto", cot=False, me=0, realme=False):
        self.llm_name = llm_name
        self.rag = rag
        self.me = me
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None
        self.realme = realme
        if rag:
            self.retrieval_system = RetrievalSystem(
                self.retriever_name, self.corpus_name, self.db_dir, cache=corpus_cache, HNSW=HNSW)
        else:
            self.retrieval_system = None
        if cot:
            self.templates = {"cot_system": general_cot_system, "cot_prompt": general_cot,
                              "medrag_system": general_medrag_system, "medrag_prompt": general_medrag}
        else:
            self.templates = {"cot_system": general_cot_system2, "cot_prompt": general_cot2,
                              "medrag_system": general_medrag_system2, "medrag_prompt": general_medrag2}
        self.templates["general_extract"] = general_extract_nolist
        if self.llm_name.split('/')[0].lower() == "openai":
            self.model = self.llm_name.split('/')[-1]
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 16384
                self.context_length = 15000
            elif "gpt-4" in self.model:
                self.max_length = 32768
                self.context_length = 30000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.max_length = 2048
            self.context_length = 1024
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_name, cache_dir=self.cache_dir)
            if "llama-2" in llm_name.lower():
                self.max_length = 4096
                self.context_length = 3072
            elif "llama-3" in llm_name.lower():
                self.max_length = 8192
                self.context_length = 7168
                if ".1" in llm_name or ".2" in llm_name:
                    self.max_length = 131072
                    self.context_length = 128000
            elif "pmc_llama" in llm_name.lower():
                self.tokenizer.chat_template = open(
                    './templates/pmc_llama.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 2048
                self.context_length = 1024
            elif "qwen" in llm_name.lower():
                self.max_length = 131072
                self.context_length = 128000

            self.model = transformers.pipeline(
                "text-generation",
                model=self.llm_name,
                # torch_dtype=torch.float16,
                torch_dtype=torch.bfloat16,
                device_map=device,
                model_kwargs={"cache_dir": self.cache_dir},
            )
            if "llama-3" in llm_name.lower():
                self.tokenizer = self.model.tokenizer

        self.follow_up = follow_up
        if self.rag and self.follow_up:
            self.answer = self.i_medrag_answer
            self.templates["medrag_system"] = simple_medrag_system
            self.templates["medrag_prompt"] = simple_medrag_prompt
            self.templates["i_medrag_system"] = i_medrag_system
            self.templates["follow_up_ask"] = follow_up_instruction_ask
            self.templates["follow_up_answer"] = follow_up_instruction_answer
        else:
            if self.realme:
                self.answer = self.medrag_answer_realme
            else:
                self.answer = self.medrag_answer

    def custom_stop(self, stop_str, input_len=0):
        stopping_criteria = StoppingCriteriaList(
            [CustomStoppingCriteria(stop_str, self.tokenizer, input_len)])
        return stopping_criteria

    def generate(self, messages):
        '''
        generate response given messages
        '''
        if "openai" in self.llm_name.lower():
            ans = openai_client(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
        elif "gemini" in self.llm_name.lower():
            response = self.model.generate_content(
                messages[0]["content"] + '\n\n' + messages[1]["content"])
            ans = response.candidates[0].content.parts[0].text
        else:
            stopping_criteria = None
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            if "meditron" in self.llm_name.lower():
                # stopping_criteria = custom_stop(["###", "User:", "\n\n\n"], self.tokenizer, input_len=len(self.tokenizer.encode(prompt_cot, add_special_tokens=True)))
                stopping_criteria = self.custom_stop(["###", "User:", "\n\n\n"], input_len=len(
                    self.tokenizer.encode(prompt, add_special_tokens=True)))
            elif "llama-3" in self.llm_name.lower():
                response = self.model(
                    prompt,
                    temperature=None,
                    top_p=None,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    # max_length=self.max_length,
                    max_new_tokens=4096,
                    repetition_penalty=1.2,
                    truncation=True,
                    stopping_criteria=None,
                )
            elif "qwen" in self.llm_name.lower():
                response = self.model(
                    prompt,
                    temperature=None,
                    top_p=None,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    # max_length=self.max_length,
                    max_new_tokens=4096,
                    repetition_penalty=1.2,
                    truncation=True,
                    stopping_criteria=None,
                )
            else:
                response = self.model(
                    prompt,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_length,
                    truncation=True,
                    stopping_criteria=stopping_criteria
                )
            # ans = response[0]["generated_text"]
            ans = response[0]["generated_text"][len(prompt):]
        return ans

    def extract_factual_info_rag(self, question, retrieved_snippets):
        num_sentences, other_sentences, last_sentence = self.split_sentences(
            question)
        contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(
            idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
        answers = []
        if len(contexts) == 0:
            contexts = [""]
        if "openai" in self.llm_name.lower():
            contexts = [self.tokenizer.decode(self.tokenizer.encode(
                "\n".join(contexts))[:self.context_length])]
        elif "gemini" in self.llm_name.lower():
            contexts = [self.tokenizer.decode(self.tokenizer.encode(
                "\n".join(contexts))[:self.context_length])]
        else:
            contexts = [self.tokenizer.decode(self.tokenizer.encode(
                "\n".join(contexts), add_special_tokens=False)[:self.context_length])]
        for context in contexts:

            prompt_extract = self.templates["general_extract"].render(
                context=context, ehr=other_sentences, question=last_sentence)
            messages = [

                {"role": "user", "content": prompt_extract}
            ]
            ans = self.generate(messages)
            answers.append(re.sub("\s+", " ", ans))
        return answers

    def extract_factual_info(self, question):
        # prompt = '''Please extract the key factual information relevant to solving this problem and present it as a Python list.
        # Use concise descriptions for each item, formatted as ["key detail 1", ..., "key detail N"].'''
        prompt = '''Please extract the key factual information relevant to solving this problem and present it as a Python list. 
        Use concise descriptions for each item, formatted as ["key detail 1", ..., "key detail N"]. For example, ['Patient age: 39 years (Middle-aged)', 'Symptoms: fever, chills, left lower quadrant abdominal pain', 'Vital signs: high temperature (39.1°C or 102.3°F), tachycardia (pulse 126/min), tachypnea (respirations 28/min) and hypotension (blood pressure 80/50 mmHg)', 'Physical exam findings: mucopurulent discharge from the cervical os and left adnexal tenderness', 'Laboratory results: low platelet count (14,200/mm^3), elevated D-dimer (965 ng/mL)', 'Phenol test result: identification of a phosphorylated N-acetylglucosame dimmer with 6 fatty acids attached to a polysaccharide side chain']'''
        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": question + "\n" + prompt},
        ]

        ans = self.generate(messages)
        answers = []
        answers.append(re.sub("\s+", " ", ans))
        answers = answers[0]

        print(f"Generated Answer: {answers}")

        matched_items = re.findall(r'"([^"]*)"', answers)

        if matched_items:
            print(f"number queries: {len(matched_items)}")
            print(f"extract info: {matched_items}")

            return matched_items, answers
        else:
            print("no info found")
            return [], answers

    def generate_possible_content(self, question):

        prompt = '''Please generate some knowledge that might address the above question. please give me only the knowledge.'''

        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": question + "\n" + prompt},
        ]
        ans = self.generate(messages)
        answers = []
        answers.append(re.sub("\s+", " ", ans))
        answers = answers[0]

        # print(f"Generated Answer: {answers}")
        return answers

    def generate_possible_answer(self, question):

        # prompt = '''Please generate some knowledge that might address the above question. please give me only the knowledge.'''
        prompt = '''Please give 4 options for the question. Each option should be a concise description of a key detail, formatted as:A. "key detail 1" B. "key detail 2" C. "key detail 3" D. "key detail 4"'''
        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": question + "\n" + prompt},
        ]

        ans = self.generate(messages)
        answers = []
        answers.append(re.sub("\s+", " ", ans))
        answers = answers[0]

        # print(f"Generated Answer: {answers}")
        return answers

    def generate_possible_title(self, question):

        prompt = '''Please generate some titles of references that might address the above question. Please give me only the titles, formatted as: ["title 1", "title 2", ..., "title N"]. Please be careful not to give specific content and analysis, just the title.'''

        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": question + "\n" + prompt},
        ]

        ans = self.generate(messages)
        answers = []
        answers.append(re.sub("\s+", " ", ans))
        answers = answers[0]

        # print(f"Generated Answer: {answers}")
        return answers

    def split_sentences(self, text):

        text = text.rstrip('"').strip()

        pattern = r'(.*?[.!?。\n])'
        sentences = re.findall(pattern, text, re.DOTALL)

        if not sentences:
            return 0, "", ""

        last_sentence = sentences[-1].strip()
        other_sentences = "".join(sentences[:-1]).strip()

        return len(sentences), other_sentences, last_sentence

    def retrieve_me_GAR_original_pro(self, question, k=32, rrf_k=100,
                                     search_method="best_of_n", n=3, expand_size=3,
                                     num_rollouts=2, beam_size=1, max_path_length=3,
                                     temperature=0.7):
        """
        Enhanced retrieval method that uses path sampling techniques (best_of_n or tree_search)
        to generate better queries for retrieval.
        
        Args:
            question: The main question to answer
            options: Optional answer choices
            k: Number of documents to retrieve
            rrf_k: Parameter for reciprocal rank fusion
            search_method: Either "best_of_n" or "tree_search"
            n: Number of paths to sample for best_of_n
            expand_size: Number of subqueries to generate per state for tree_search
            num_rollouts: Number of rollouts per state for tree_search
            beam_size: Beam size for tree_search
            max_path_length: Maximum path length for search
            temperature: Temperature for sampling
            
        Returns:
            all_retrieved_snippets: List of retrieved document snippets
            all_scores: Corresponding relevance scores
        """

        _, other_sentences, last_sentence = self.split_sentences(
            question)
        task_desc = "Retrieve relevant medical information to help answer this question"
        if other_sentences == "":
            original_answers = ""
        else:
            _, original_answers = self.extract_factual_info(question)

        if search_method == "best_of_n":
            path = self.best_of_n(
                query=original_answers+last_sentence,
                task_desc=task_desc,
                max_path_length=max_path_length,
                temperature=temperature,
                n=n,
                k=k, rrf_k=rrf_k
            )
        elif search_method == "tree_search":
            path = self.tree_search(
                query=original_answers+last_sentence,
                task_desc=task_desc,
                max_path_length=max_path_length,
                temperature=temperature,
                expand_size=expand_size,
                num_rollouts=num_rollouts,
                beam_size=beam_size,
                k=k, rrf_k=rrf_k
            )
        else:
            raise ValueError(f"Unknown search method: {search_method}")

        return path
        

    def retrieve_me_GAR_original(self, question, options="", k=32, rrf_k=100):

        _, _ , last_sentence = self.split_sentences(
            question)
        quarter_k = k // 4
        all_retrieved_snippets = []
        all_scores = []
        retrieved_snippets, scores = self.retrieval_system.retrieve(
            last_sentence, k=quarter_k, rrf_k=rrf_k)
        all_retrieved_snippets.extend(retrieved_snippets)
        all_scores.extend(scores)

        options = '\n'.join([key+". "+options[key]
                            for key in sorted(options.keys())])
        retrieved_snippets, scores = self.retrieval_system.retrieve(
            options, k=quarter_k, rrf_k=rrf_k)
        all_retrieved_snippets.extend(retrieved_snippets)
        all_scores.extend(scores)

        possible_content = self.generate_possible_content(question)
        retrieved_snippets, scores = self.retrieval_system.retrieve(
            possible_content, k=quarter_k, rrf_k=rrf_k)
        all_retrieved_snippets.extend(retrieved_snippets)
        all_scores.extend(scores)

        possible_title = self.generate_possible_title(question)
        retrieved_snippets, scores = self.retrieval_system.retrieve(
            possible_title, k=quarter_k, rrf_k=rrf_k)
        all_retrieved_snippets.extend(retrieved_snippets)
        all_scores.extend(scores)

        return all_retrieved_snippets, all_scores

    def medrag_answer_realme(self, question, options=None, k=32, rrf_k=100, save_dir=None, snippets=None, snippets_ids=None, num_rounds=2, **kwargs):
        '''
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): number of snippets to retrieve
        rrf_k (int): parameter for Reciprocal Rank Fusion
        save_dir (str): directory to save the results
        snippets (List[Dict]): list of snippets to be used
        snippets_ids (List[Dict]): list of snippet ids to be used
        '''
        options = '\n'.join([key+". "+options[key]
                            for key in sorted(options.keys())])

        retrieved_snippets, scores = self.retrieval_system.retrieve(
            question, k=k, rrf_k=rrf_k)
        all_retrieved_snippets = retrieved_snippets
        all_scores = scores
        for i in range(num_rounds):
            # extract factual information
            num_sentences, other_sentences, last_sentence = self.split_sentences(
                question)
            if other_sentences == "":
                extract_sentences = ""
            else:
                extract_sentences = self.extract_factual_info_rag(
                    question, all_retrieved_snippets)
                extract_sentences = str(extract_sentences)
                # print(extract_answers)
            half_k = k // 2
            quarter_k = k // 4
            all_retrieved_snippets = []
            all_scores = []
            # GAR
            possible_answers = self.generate_possible_answer(question)
            print(possible_answers)
            retrieved_snippets, scores = self.retrieval_system.retrieve(
                question+possible_answers, k=half_k, rrf_k=rrf_k)
            all_retrieved_snippets.extend(retrieved_snippets)

            possible_content = self.generate_possible_content(question)
            print(possible_content)
            retrieved_snippets, scores = self.retrieval_system.retrieve(
                possible_content+question, k=quarter_k, rrf_k=rrf_k)
            all_retrieved_snippets.extend(retrieved_snippets)
            all_scores.extend(scores)

            possible_title = self.generate_possible_title(question)
            print(possible_title)
            retrieved_snippets, scores = self.retrieval_system.retrieve(
                possible_title+question, k=quarter_k, rrf_k=rrf_k)
            all_retrieved_snippets.extend(retrieved_snippets)
            all_scores.extend(scores)

        contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(
            idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
        if len(contexts) == 0:
            contexts = [""]
        if "openai" in self.llm_name.lower():
            contexts = [self.tokenizer.decode(self.tokenizer.encode(
                "\n".join(contexts))[:self.context_length])]
        elif "gemini" in self.llm_name.lower():
            contexts = [self.tokenizer.decode(self.tokenizer.encode(
                "\n".join(contexts))[:self.context_length])]
        else:
            contexts = [self.tokenizer.decode(self.tokenizer.encode(
                "\n".join(contexts), add_special_tokens=False)[:self.context_length])]

        # generate answers
        answers = []
        for context in contexts:
            prompt_medrag = self.templates["medrag_prompt"].render(
                context=context, question=question, options=options)
            messages = [
                {"role": "system", "content": self.templates["medrag_system"]},
                {"role": "user", "content": prompt_medrag}
            ]
            ans = self.generate(messages)
            print(ans)
            messages.append({"role": "assistant", "content": ans})
            messages.append({"role": "user", "content": "Options:\n"+options +
                            "\n Output the answer in JSON: {'answer': your_answer (A/B/C/D)}"})
            ans = self.generate(messages)
            print(ans)
            answers.append(re.sub("\s+", " ", ans))

        return answers[0] if len(answers) == 1 else answers, retrieved_snippets, scores

    def medrag_answer(self, question, options=None, k=32, rrf_k=100, save_dir=None, snippets=None, snippets_ids=None, **kwargs):
        '''
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): number of snippets to retrieve
        rrf_k (int): parameter for Reciprocal Rank Fusion
        save_dir (str): directory to save the results
        snippets (List[Dict]): list of snippets to be used
        snippets_ids (List[Dict]): list of snippet ids to be used
        '''

        copy_options = options
        if options is not None:
            options = '\n'.join([key+". "+options[key]
                                for key in sorted(options.keys())])
        else:
            options = ''

        # retrieve relevant snippets
        if self.rag:
            if snippets is not None:
                retrieved_snippets = snippets[:k]
                scores = []
            elif snippets_ids is not None:
                if self.docExt is None:
                    self.docExt = DocExtracter(
                        db_dir=self.db_dir, cache=True, corpus_name=self.corpus_name)
                retrieved_snippets = self.docExt.extract(snippets_ids[:k])
                scores = []
            else:
                assert self.retrieval_system is not None
                # No transform for the question
                if self.me == 0:
                    retrieved_snippets, scores = self.retrieval_system.retrieve(
                        question, k=k, rrf_k=rrf_k)
                # transform the question by GAR, can be seen as round 0
                elif self.me == 1:
                    retrieved_snippets, scores = self.retrieve_me_GAR_original(
                        question, copy_options, k, rrf_k)
                # transform the question by RGAR, easy implementation for round 1
                elif self.me == 2:
                    #TODO: alter data structure to accept the path data structure!!
                    retrieved_snippets, scores = self.retrieve_me_GAR_original_pro(
                        question, copy_options, k, rrf_k)

            contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(
                idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
            if len(contexts) == 0:
                contexts = [""]
            if "openai" in self.llm_name.lower():
                contexts = [self.tokenizer.decode(self.tokenizer.encode(
                    "\n".join(contexts))[:self.context_length])]
            else:
                contexts = [self.tokenizer.decode(self.tokenizer.encode(
                    "\n".join(contexts), add_special_tokens=False)[:self.context_length])]
        else:
            retrieved_snippets = []
            scores = []
            contexts = []

        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # generate answers
        answers = []
        if not self.rag:
            prompt_cot = self.templates["cot_prompt"].render(
                question=question, options=options)
            messages = [
                {"role": "system", "content": self.templates["cot_system"]},
                {"role": "user", "content": prompt_cot}
            ]
            ans = self.generate(messages)
            answers.append(re.sub("\s+", " ", ans))
        else:
            for context in contexts:
                prompt_medrag = self.templates["medrag_prompt"].render(
                    context=context, question=question, options=options)
                messages = [
                    {"role": "system",
                        "content": self.templates["medrag_system"]},
                    {"role": "user", "content": prompt_medrag}
                ]
                ans = self.generate(messages)
                print(ans)
                answers.append(re.sub("\s+", " ", ans))

        if save_dir is not None:
            with open(os.path.join(save_dir, "snippets.json"), 'w') as f:
                json.dump(retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response.json"), 'w') as f:
                json.dump(answers, f, indent=4)

        return answers[0] if len(answers) == 1 else answers, retrieved_snippets, scores

    def sample_path(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            k: int = 32, rrf_k: int = 100,
            **kwargs
    ) -> RagPath:
        past_subqueries: List[str] = kwargs.pop('past_subqueries', [])
        past_subanswers: List[str] = kwargs.pop('past_subanswers', [])
        past_doc_ids: List[List[str]] = kwargs.pop('past_doc_ids', [])
        assert len(past_subqueries) == len(past_subanswers) == len(past_doc_ids)

        subquery_temp: float = temperature
        num_llm_calls: int = 0
        max_num_llm_calls: int = 3 * (max_path_length - len(past_subqueries))

        # from RGAR
        half_k = k // 2
        quarter_k = k // 4
        all_retrieved_snippets = []
        all_scores = []

        while len(past_subqueries) < max_path_length and num_llm_calls < max_num_llm_calls:
            num_llm_calls += 1
            messages: List[Dict] = get_generate_subquery_prompt(
                query=query,
                past_subqueries=past_subqueries,
                past_subanswers=past_subanswers,
                task_desc=task_desc,
            )
            possible_content = self.generate_possible_content(query)
            retrieved_snippets, scores = self.retrieval_system.retrieve(
                possible_content, k=quarter_k, rrf_k=rrf_k)
            all_retrieved_snippets.extend(retrieved_snippets)
            all_scores.extend(scores)

            possible_title = self.generate_possible_title(query)
            retrieved_snippets, scores = self.retrieval_system.retrieve(
                possible_title, k=quarter_k, rrf_k=rrf_k)
            all_retrieved_snippets.extend(retrieved_snippets)
            all_scores.extend(scores)

            possible_answers = self.generate_possible_answer(query)
            retrieved_snippets, scores = self.retrieval_system.retrieve(
                query+possible_answers, k=half_k, rrf_k=rrf_k)
            all_retrieved_snippets.extend(retrieved_snippets)
            all_scores.extend(scores)

            return all_retrieved_snippets, all_scores

            subquery: str = self.generate(messages=messages)
            subquery = _normalize_subquery(subquery)

            if subquery in past_subqueries:
                subquery_temp = max(subquery_temp, 0.7)
                continue

            subquery_temp = temperature
            subanswer, doc_ids = self._get_subanswer_and_doc_ids(
                subquery=subquery, max_message_length=max_message_length
            )

            past_subqueries.append(subquery)
            past_subanswers.append(subanswer)
            past_doc_ids.append(doc_ids)

        return RagPath(
            query=query,
            past_subqueries=past_subqueries,
            past_subanswers=past_subanswers,
            past_doc_ids=past_doc_ids,
        )

    def generate_final_answer(
            self, corag_sample: RagPath, task_desc: str,
            max_message_length: int = 4096,
            documents: Optional[List[str]] = None, **kwargs
    ) -> str:
        messages: List[Dict] = get_generate_final_answer_prompt(
            query=corag_sample.query,
            past_subqueries=corag_sample.past_subqueries or [],
            past_subanswers=corag_sample.past_subanswers or [],
            task_desc=task_desc,
            documents=documents,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        return self.generate(messages=messages)

    def _truncate_long_messages(self, messages: List[Dict], max_length: int):
        for msg in messages:
            if len(msg['content']) < 2 * max_length:
                continue

            with self.lock:
                msg['content'] = batch_truncate(
                    [msg['content']], tokenizer=self.tokenizer, max_length=max_length, truncate_from_middle=True
                )[0]

    def sample_subqueries(self, query: str, task_desc: str, n: int = 10, max_message_length: int = 4096, **kwargs) -> List[str]:
        messages: List[Dict] = get_generate_subquery_prompt(
            query=query,
            past_subqueries=kwargs.pop('past_subqueries', []),
            past_subanswers=kwargs.pop('past_subanswers', []),
            task_desc=task_desc,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        completion = self.generate(messages=messages)
        subqueries = [_normalize_subquery(completion)]
        subqueries = list(set(subqueries))[:n]

        return subqueries

    def _get_subanswer_and_doc_ids(
            self, subquery: str, max_message_length: int = 4096
    ) -> Tuple[str, List]:
        # TODO: get scores??
        retriever_results, scores = self.retrieval_system.retrieve(
            query=subquery, k=5, rrf_k=100
        )
        doc_ids: List[str] = [res['id'] for res in retriever_results]
        documents: List[str] = [format_input_context(self.corpus[int(doc_id)]) for doc_id in doc_ids][::-1]

        messages: List[Dict] = get_generate_intermediate_answer_prompt(
            subquery=subquery,
            documents=documents,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        subanswer: str = self.generate(messages=messages)
        return subanswer, doc_ids

    def tree_search(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            expand_size: int = 4, num_rollouts: int = 2, beam_size: int = 1,
            k: int =32, rrf_k: int = 100,
            **kwargs
    ) -> RagPath:
        return self._search(
            query=query, task_desc=task_desc,
            max_path_length=max_path_length,
            max_message_length=max_message_length,
            temperature=temperature,k=k,rrf_k=rrf_k,
            expand_size=expand_size, num_rollouts=num_rollouts, beam_size=beam_size,
            **kwargs
        )

    def best_of_n(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            n: int = 4,
            k: int =32, rrf_k: int = 100,
            **kwargs
    ) -> RagPath:
        sampled_paths: List[RagPath] = []
        for idx in range(n):
            path: RagPath = self.sample_path(
                query=query, task_desc=task_desc,
                max_path_length=max_path_length,
                max_message_length=max_message_length,
                temperature=0. if idx == 0 else temperature,
                k=k, rrf_k=rrf_k,
                **kwargs
            )
            sampled_paths.append(path)

        scores: List[float] = [self._eval_single_path(p) for p in sampled_paths]
        return sampled_paths[scores.index(min(scores))]

    def _search(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            expand_size: int = 4, num_rollouts: int = 2, beam_size: int = 1,
            **kwargs
    ) -> RagPath:
        candidates: List[RagPath] = [RagPath(query=query, past_subqueries=[], past_subanswers=[], past_doc_ids=[])]
        for step in range(max_path_length):
            new_candidates: List[RagPath] = []
            for candidate in candidates:
                new_subqueries: List[str] = self.sample_subqueries(
                    query=query, task_desc=task_desc,
                    past_subqueries=deepcopy(candidate.past_subqueries),
                    past_subanswers=deepcopy(candidate.past_subanswers),
                    n=expand_size, temperature=temperature, max_message_length=max_message_length
                )
                for subquery in new_subqueries:
                    new_candidate: RagPath = deepcopy(candidate)
                    new_candidate.past_subqueries.append(subquery)
                    subanswer, doc_ids = self._get_subanswer_and_doc_ids(
                        subquery=subquery, max_message_length=max_message_length
                    )
                    new_candidate.past_subanswers.append(subanswer)
                    new_candidate.past_doc_ids.append(doc_ids)
                    new_candidates.append(new_candidate)

            if len(new_candidates) > beam_size:
                scores: List[float] = []
                for path in new_candidates:
                    score = self._eval_state_without_answer(
                        path, num_rollouts=num_rollouts,
                        task_desc=task_desc,
                        max_path_length=max_path_length,
                        temperature=temperature,
                        max_message_length=max_message_length
                    )
                    scores.append(score)

                # lower scores are better
                new_candidates = [c for c, s in sorted(zip(new_candidates, scores), key=lambda x: x[1])][:beam_size]

            candidates = new_candidates

        return candidates[0]

    def _eval_single_path(self, current_path: RagPath, max_message_length: int = 4096) -> float:
        messages: List[Dict] = get_generate_intermediate_answer_prompt(
            subquery=current_path.query,
            documents=[f'Q: {q}\nA: {a}' for q, a in zip(current_path.past_subqueries, current_path.past_subanswers)],
        )
        messages.append({'role': 'assistant', 'content': 'No relevant information found'})
        self._truncate_long_messages(messages, max_length=max_message_length)

        # TODO: call llama agent from RGAR instead
        response = self.generate(messages=messages)
        answer_logprobs: List[float] = parse_answer_logprobs(response)

        return sum(answer_logprobs) / len(answer_logprobs)

    def _eval_state_without_answer(
            self, path: RagPath, num_rollouts: int, task_desc: str,
            max_path_length: int = 3,
            temperature: float = 0.7,
            max_message_length: int = 4096
    ) -> float:
        assert len(path.past_subqueries) > 0
        if num_rollouts <= 0:
            return self._eval_single_path(path)

        rollout_paths: List[RagPath] = []
        for _ in range(num_rollouts):
            rollout_path: RagPath = self.sample_path(
                query=path.query, task_desc=task_desc,
                max_path_length=min(max_path_length, len(path.past_subqueries) + 2),  # rollout at most 2 steps
                temperature=temperature, max_message_length=max_message_length,
                past_subqueries=deepcopy(path.past_subqueries),
                past_subanswers=deepcopy(path.past_subanswers),
                past_doc_ids=deepcopy(path.past_doc_ids),
            )
            rollout_paths.append(rollout_path)

        scores: List[float] = [self._eval_single_path(p) for p in rollout_paths]
        return sum(scores) / len(scores)
    

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(stop in tokens for stop in self.stops_words)
