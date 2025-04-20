# This file is adapted from [Teddy-XiongGZ/MedRAG]
# Original source: [https://github.com/Teddy-XiongGZ/MedRAG/blob/main/src/medrag.py]
# we developed RGAR based on MedRAG
# we developed RAG systems without CoT based on MedRAG
# we add support for qwens
from utils import batch_truncate
from agent_utils import RagPath
from prompts import get_generate_subquery_prompt, get_generate_intermediate_answer_prompt, get_generate_final_answer_prompt
from data_utils import format_input_context, parse_answer_logprobs
from typing import Optional, List, Dict, Tuple
from copy import deepcopy
import tiktoken
from transformers import StoppingCriteria, StoppingCriteriaList
import openai
from transformers import AutoTokenizer
import transformers
import torch
import json
import re
import os
from utils import RetrievalSystem, DocExtracter
from template import *
from config import config
import sys
sys.path.append("src")

# imports for added CoRAG methods


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

    def generate(self, messages, return_logprobs=False):
        '''
        generate response given messages
        '''
        if "openai" in self.llm_name.lower():
            ans = openai_client(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            return ans
        elif "gemini" in self.llm_name.lower():
            response = self.model.generate_content(
                messages[0]["content"] + '\n\n' + messages[1]["content"])
            ans = response.candidates[0].content.parts[0].text
            return ans
        else:
            stopping_criteria = None
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            
            if "meditron" in self.llm_name.lower():
                stopping_criteria = self.custom_stop(["###", "User:", "\n\n\n"], input_len=len(
                    self.tokenizer.encode(prompt, add_special_tokens=True)))
            elif "llama-3" in self.llm_name.lower():
                # For llama-3.2, we can get log probabilities
                if return_logprobs:
                    response = self.model(
                        prompt,
                        temperature=None,
                        top_p=None,
                        do_sample=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        max_new_tokens=4096,
                        repetition_penalty=1.2,
                        truncation=True,
                        stopping_criteria=None,
                        output_scores=True,  # Request log probabilities
                        return_dict_in_generate=True
                    )
                    # Extract log probabilities
                    logprobs = []
                    for token_scores in response.scores:
                        # Get the log probability of the chosen token
                        logprob = token_scores[0].max().item()
                        logprobs.append(logprob)
                    
                    ans = response.sequences[0][len(self.tokenizer.encode(prompt)):]
                    ans = self.tokenizer.decode(ans, skip_special_tokens=True)
                    return ans, logprobs
                else:
                    response = self.model(
                        prompt,
                        temperature=None,
                        top_p=None,
                        do_sample=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
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
            
            ans = response[0]["generated_text"][len(prompt):]
            return ans

    def extract_factual_info_rag(self, question, retrieved_snippets):
        _, other_sentences, last_sentence = self.split_sentences(
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

    def extract_factual_info(self, question, context):
        # prompt = '''Please extract the key factual information relevant to solving this problem and present it as a Python list.
        # Use concise descriptions for each item, formatted as ["key detail 1", ..., "key detail N"].'''
        prompt = '''Please extract the key factual information relevant to addressing the question and present it as a Python list. 
        Use concise descriptions for each item, formatted as ["key detail 1", ..., "key detail N"]. For example, ['Patient age: 39 years (Middle-aged)', 'Symptoms: fever, chills, left lower quadrant abdominal pain', 'Vital signs: high temperature (39.1°C or 102.3°F), tachycardia (pulse 126/min), tachypnea (respirations 28/min) and hypotension (blood pressure 80/50 mmHg)', 'Physical exam findings: mucopurulent discharge from the cervical os and left adnexal tenderness', 'Laboratory results: low platelet count (14,200/mm^3), elevated D-dimer (965 ng/mL)', 'Phenol test result: identification of a phosphorylated N-acetylglucosame dimmer with 6 fatty acids attached to a polysaccharide side chain']'''
        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": context + "\n" + prompt + "\n Question:" + question},
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

        print(f"Generated Answer: {answers}")
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

        print(f"Generated Answer: {answers}")
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

        print(f"Generated Answer: {answers}")
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
                                      n=3, max_path_length=3, temperature=0.7):
        """
        Enhanced retrieval method that uses path sampling techniques (best_of_n or tree_search)
        to generate better queries for retrieval.
        """

        # last sentence is always the question, prior info is patient record info
        _, patient_context, patient_question = self.split_sentences(
            question)

        path, retrieved_snippets = self.best_of_n(
            query=patient_question,
            context=patient_context,
            max_path_length=max_path_length,
            temperature=temperature,
            n=n,
            k=k, rrf_k=rrf_k
        )

        return path, retrieved_snippets

    def retrieve_me_GAR_original(self, question, options="", k=32, rrf_k=100):

        _, _, last_sentence = self.split_sentences(
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
                    # now returning the best path!
                    best_path, retrieved_snippets = self.retrieve_me_GAR_original_pro(
                        question, k, rrf_k)

            contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(
                idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
            if len(contexts) == 0:
                contexts = [""]

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
                prompt_medrag = get_generate_final_answer_prompt(best_path.query, best_path.past_subqueries, best_path.past_subanswers, context, options)
                # prompt_medrag = self.templates["medrag_prompt"].render(
                #     context=context, question=question, options=options)
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
            self, query: str, context: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            k: int = 32, rrf_k: int = 100,
            **kwargs
    ) -> Tuple[RagPath, List[Dict]]:
        past_subqueries: List[str] = kwargs.pop('past_subqueries', [])
        past_subanswers: List[str] = kwargs.pop('past_subanswers', [])
        scores: List[float] = kwargs.pop('scores', [])
        assert len(past_subqueries) == len(past_subanswers) == len(scores)

        # For a single path, generate max_path_length subqueries and subanswers
        num_llm_calls: int = 0
        max_num_llm_calls: int = 3 * (max_path_length - len(past_subqueries))

        # from RGAR
        all_retrieved_snippets = []
        all_scores = []

        # get relevant patient facts formatted
        patient_facts, _ = self.extract_factual_info(query, context)

        # generate possible medical titles and content
        possible_content = self.generate_possible_content(query)
        retrieved_snippets, scores = self.retrieval_system.retrieve(
            possible_content, k=k//4, rrf_k=rrf_k)
        all_retrieved_snippets.extend(retrieved_snippets)
        all_scores.extend(scores)
        
        possible_titles = self.generate_possible_title(query)
        retrieved_snippets, scores = self.retrieval_system.retrieve(
            possible_titles, k=k//4, rrf_k=rrf_k)
        all_retrieved_snippets.extend(retrieved_snippets)
        all_scores.extend(scores)

        # generate initial set of documents
        documents = ["Document [{:d}] (Title: {:s}) {:s}".format(
            idx, snippet["title"], snippet["content"]) 
            for idx, snippet in enumerate(all_retrieved_snippets)]

        while len(past_subqueries) < max_path_length and num_llm_calls < max_num_llm_calls:
            num_llm_calls += 1

            # generate subquery 
            subquery_messages: List[Dict] = get_generate_subquery_prompt(
                query=query, 
                past_subqueries=past_subqueries,
                past_subanswers=past_subanswers,
                extracted_facts=patient_facts
            )
            subquery = self.generate(messages=subquery_messages)
            subquery = subquery.strip()

            # skip if seen before
            if subquery in past_subqueries:
                continue

            subquery_snippets, _ = self.retrieval_system.retrieve(
                query=subquery, k=k//4, rrf_k=rrf_k)

            # Format documents for the intermediate answer generation
            subquery_documents = ["Document [{:d}] (Title: {:s}) {:s}".format(
            idx + len(all_retrieved_snippets), snippet["title"], snippet["content"]) 
            for idx, snippet in enumerate(subquery_snippets)]

            combined_docs = documents.copy()
            combined_docs.extend(subquery_documents)

            messages: List[Dict] = get_generate_intermediate_answer_prompt(
                subquery=subquery,
                documents=combined_docs)
            
            # Get both answer and log probabilities in one call
            subanswer, logprobs = self.generate(messages=messages, return_logprobs=True)
            
            # Store the log probabilities directly in the path
            if logprobs:
                scores.append(sum(logprobs) / len(logprobs))
            else:
                scores.append(0.0)

            past_subqueries.append(subquery)
            past_subanswers.append(subanswer)

        return (
            RagPath(
                query=query,
                past_subqueries=past_subqueries,
                past_subanswers=past_subanswers,
                scores=scores
            ),
            all_retrieved_snippets
        )

    def generate_final_answer(
            self, corag_sample: RagPath, context: str,
            max_message_length: int = 4096,
            documents: Optional[List[str]] = None, **kwargs
    ) -> str:
        messages: List[Dict] = get_generate_final_answer_prompt(
            query=corag_sample.query,
            past_subqueries=corag_sample.past_subqueries or [],
            past_subanswers=corag_sample.past_subanswers or [],
            context=context,
            documents=documents,
        )
        # self._truncate_long_messages(messages, max_length=max_message_length)

        return self.generate(messages=messages)

    # def _truncate_long_messages(self, messages: List[Dict], max_length: int):
    #     for msg in messages:
    #         if len(msg['content']) < 2 * max_length:
    #             continue
            
    #         #TODO: get batch_truncate back!!
    #         with self.lock:
    #             msg['content'] = batch_truncate(
    #                 [msg['content']], tokenizer=self.tokenizer, max_length=max_length, truncate_from_middle=True
    #             )[0]

    def best_of_n(
            self, query: str, context: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            n: int = 4,
            k: int = 32, rrf_k: int = 100,
            **kwargs
    ) -> RagPath:
        sampled_paths: List[RagPath] = []
        for idx in range(n):
            path, retrieved_snippets = self.sample_path(
                query=query, context=context,
                max_path_length=max_path_length,
                max_message_length=max_message_length,
                temperature=0. if idx == 0 else temperature,
                k=k, rrf_k=rrf_k,
                **kwargs
            )
            sampled_paths.append(path)

        scores: List[float] = [self._eval_single_path(p) for p in sampled_paths]
        return sampled_paths[scores.index(min(scores))], retrieved_snippets

    def _eval_single_path(self, current_path: RagPath, max_message_length: int = 4096) -> float:
        # Use scores directly from the path if available
        if current_path.scores:
            return sum(current_path.scores) / len(current_path.scores)
        
        # # Fallback to generating new scores if not available
        # messages: List[Dict] = get_generate_intermediate_answer_prompt(
        #     subquery=current_path.query,
        #     documents=[f'Q: {q}\nA: {a}' for q, a in zip(current_path.past_subqueries, current_path.past_subanswers)],
        # )
        # messages.append({'role': 'assistant', 'content': 'No relevant information found'})

        # # Get both answer and log probabilities
        # answer, logprobs = self.generate(messages=messages, return_logprobs=True)
        
        # # Calculate average log probability
        # if logprobs:
        #     return sum(logprobs) / len(logprobs)
        # else:
        #     # Fallback if no log probabilities available
        #     return 0.0


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(stop in tokens for stop in self.stops_words)
