from arxiv2026_instruction_vectors import config
from datasets import Dataset, load_dataset, disable_caching
import os
import numpy as np
import pandas as pd
import string
import torch.utils.data

disable_caching()


class BigBenchTask:
    def __init__(self,
                 task_name,
                 tokenizer,
                 sample_start=0,
                 num_samples=None,
                 context_prompt_idx=0,
                 cache_prompt_idx=None,
                 local=True,
                 ):
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.sample_start = sample_start
        self.num_samples = num_samples
        self.context_prompt_idx = context_prompt_idx
        self.cache_prompt_idx = cache_prompt_idx
        self.instruction_prompt = self.get_instruction_prompt()
        self.adjacent_words = self.get_adjacent_words()
        self.boilerplate_words = self.get_boilerplate_words()
        self.varied_instructions = self.get_varied_instructions()
        self.varied_labels = self.get_varied_label_space()
        self.local=local
        
    def get_instruction_prompt(self):
        instruction_prompts = {
            "implicatures": "Predict whether Speaker 2's answer to Speaker 1 counts as a yes or as a no. Print Y for yes and N for no. ",
            "irony_identification": "Identify whether the sentence is ironic or not. Print I for ironic and N for non-ironic. ",
            "metaphor_boolean": "For a given metaphorical sentence, identify if the second sentence is the correct interpretation. Print Y for yes and N for no. ",  # metaphor_boolean
            "object_counting": "For the given sentence, count the number of objects listed and print it as a digit. ", # object_counting
            "snarks": "Choose which of the two statements is sarcastic and print the corresponding letter: ", 
            "nonsense_words_grammar": "",
            "strange_stories": "",
            }
        return instruction_prompts[self.task_name]
    
    def get_varied_instructions(self):
        data_dir = os.path.join(os.path.dirname(__file__), "local_tasks", "varied_instructions", f"{self.task_name}_rephrased.csv")
        df = pd.read_csv(data_dir)
        d = Dataset.from_pandas(df)

        return d["instruction"]
    
    def get_varied_label_space(self):
        varied_label_space = {
            "implicatures": [
                "Predict whether Speaker 2's answer to Speaker 1 counts as a yes or as a no. Print X for yes and Z for no. ",
                "Predict whether Speaker 2's answer to Speaker 1 counts as a yes or as a no. Print 'a' for yes and 'b' for no. ",
                "Predict whether Speaker 2's answer to Speaker 1 counts as a yes or as a no. Print 'yay' for yes and 'boo' for no. ",
                "Predict whether Speaker 2's answer to Speaker 1 counts as a yes or as a no. Print 'H' for yes and 'J' for no. ",
                "Predict whether Speaker 2's answer to Speaker 1 counts as a yes or as a no. Print 'S' for yes and 'O' for no. ",
                "Predict whether Speaker 2's answer to Speaker 1 counts as a yes or as a no. Print 'yes' or 'no'.",
                "Predict whether Speaker 2's answer to Speaker 1 counts as a yes or as a no. Print 'affirmative' or 'negative'.",
                ],

            "irony_identification": [],

            "metaphor_boolean": [

                "For a given metaphorical sentence, identify if the second sentence is the correct interpretation. Print Y for yes and N for no. ",
                "For a given metaphorical sentence, identify if the second sentence is the correct interpretation. Print 'a' for yes and 'b' for no. ",
                "For a given metaphorical sentence, identify if the second sentence is the correct interpretation. Print 'yay' for yes and 'boo' for no. ",
                "For a given metaphorical sentence, identify if the second sentence is the correct interpretation. Print 'H' for yes and 'J' for no. ",
                "For a given metaphorical sentence, identify if the second sentence is the correct interpretation. Print 'S' for yes and 'O' for no. ",
                "For a given metaphorical sentence, identify if the second sentence is the correct interpretation. Print 'yes' or 'no'.",
                "For a given metaphorical sentence, identify if the second sentence is the correct interpretation. Print 'correct' or 'incorrect'.",
                "For a given metaphorical sentence, identify if the second sentence is the correct interpretation. Print 'good' or 'bad'.",
                "For a given metaphorical sentence, identify if the second sentence is the correct interpretation. Print 'right' or 'wrong'.",

                ],

            "object_counting": [
                
                "For the given sentence, count the number of objects listed and print it as a digit. ",

            ], 
            "snarks": [],
            "nonsense_words_grammar": [],
            "strange_stories": [],
            }
        return varied_label_space[self.task_name]
    
    def get_boilerplate_words(self):
        return ["Speaker", "The", "What", "Sure", "Options", "Please"]
    
    def get_adjacent_words(self):
        if self.task_name in ["antonyms", "snarks", "nonsense_words_grammar"]:
            words = []
        elif self.task_name == "implicatures":
            words1 = [w for w in string.ascii_uppercase]
            words2 = ["yes", "no", "Yes", "No"]
            words = words1 + words2
        elif self.task_name == "metaphor_boolean":
            words1 = [w for w in string.ascii_uppercase]
            words2 = ["yes", "no", "Yes", "No", "correct", "incorrect", "right", "wrong"]
            words = words1 + words2        
        elif self.task_name == "object_counting":
            words1 = [d for d in string.digits]
            words = words1  
        return words
    
    def load_cache_prompts(self):
        prompts_for_cache = [
            "",  # Cache gets instruction only
        ]
        cache_prompts = [self.instruction_prompt.strip(" ") + prompt for prompt in prompts_for_cache]
        if self.cache_prompt_idx is not None:
            cache_prompts = [cache_prompts[self.cache_prompt_idx]]
        return cache_prompts
    
    def _preprocess_implicatures(self, example):
        prompt = example["inputs"].replace("Does Speaker 2's answer mean yes or no? \n\nQ: ", "").replace(" \nA:", " Answer:")
        example["prompt"] = prompt
        example["query"] = example["prompt"].replace("Answer:", "").rstrip(" ")
        options = ["Y", "N"]
        example["mod_label"] = options[example["multiple_choice_scores"].index(1)]
        example["mod_options"] = options
        example["instr+target"] = self.instruction_prompt + example["prompt"]
        example["instr"] = self.instruction_prompt
        return example

    def _preprocess_irony(self, example):
        prompt = example["inputs"].replace("\nIronic/Not ironic?", " Answer:")
        prompt = prompt.replace("Are the given example sentences examples of irony or not? Respond 'ironic' or 'not ironic' to each example.\n\nExample: ", "Sentence: ")
        example["prompt"] = prompt
        options = ["I", "N"]
        example["mod_label"] = options[example["multiple_choice_scores"].index(1)]
        example["mod_options"] = options
        example["instr+target"] = self.instruction_prompt + example["prompt"]
        return example
    
    def _preprocess_metaphor(self, example):
        prompt = example["inputs"].replace("The essence of the task: Given a metaphoric sentence, identify if the second sentence is the correct paraphrase of the first.\nQ: ", "Sentence 1: ")
        prompt = prompt.replace(" <-->  ", " Sentence 2: ").replace("\n  choice: True\n  choice: False\nA:", " Answer:")
        example["prompt"] = prompt
        example["query"] = example["prompt"].replace("Answer:", "").rstrip(" ")
        options = ["Y", "N"]
        example["mod_label"] = options[example["multiple_choice_scores"].index(1)]
        example["mod_options"] = options
        example["instr+target"] = self.instruction_prompt + example["prompt"]
        example["instr"] = self.instruction_prompt.rstrip(" ")
        return example

    def _preprocess_object_counting(self, example):
        prompt = example["inputs"].replace("Q: ", "Sentence: ").replace("\nA:", " Answer:")
        example["prompt"] = prompt
        example["query"] = example["prompt"].replace("Answer:", "").rstrip(" ")
        example["mod_label"] = int(example["targets"][1])  # get the numeral
        options = np.random.choice(example["mod_label"], 3)
        options = np.append(example["mod_label"], options)
        options = [str(opt) for opt in options]
        example["mod_options"] = options
        example["instr+target"] = self.instruction_prompt + example["prompt"]
        example["mod_label"] = str(example["mod_label"])
        example["instr"] = self.instruction_prompt
        return example

    def _preprocess_snarks(self, example):
        prompt = example["inputs"].replace("Q: Which statement is sarcastic? ", "")
        prompt = prompt.replace("\nA:", " Answer:")
        prompt = prompt.replace("(a)", "A: ").replace("(b)", "B: ")
        example["prompt"] = prompt
        example["query"] = example["prompt"].replace("Answer:", "").rstrip(" ")
        options = ["A", "B"]
        example["mod_label"] = options[example["multiple_choice_scores"].index(1)]
        example["mod_options"] = options
        example["instr+target"] = self.instruction_prompt + example["prompt"]
        example["instr"] = self.instruction_prompt
        return example
    
    def load_test(self):
        if self.local:
            path = "parquet"
            data_dir = f"local_tasks/{self.task_name}"
        else:
            path = "tasksource/bigbench"
            data_dir = None
        args_dict = {"path": path,
                     "data_dir": data_dir, 
                     "split": "train", 
                     "name": self.task_name,
                     }
        d = load_dataset(**args_dict)
        if self.task_name == "implicatures":
            d = d.map(self._preprocess_implicatures)
        elif self.task_name == "irony_identification":
            d = d.map(self._preprocess_irony)
        elif self.task_name == "metaphor_boolean":
            d = d.map(self._preprocess_metaphor)
        elif self.task_name == "object_counting":
            d = d.map(self._preprocess_object_counting)
        elif self.task_name == "snarks":
            d = d.map(self._preprocess_snarks)
        if self.num_samples is not None:
            d = d.select(range(0, self.num_samples))
        return d


class CsvTask:
    def __init__(self,
                 base_query,
                 subtask,
                 num_samples=None,
                 context_prompt_idx=0,
                 cache_prompt_idx=None,
                 ):
        self.base_query = base_query   # "animals" or "objects", etc. names of the csv files
        self.subtask = subtask     # "anim_color" or "can_fly"
        self.num_samples = num_samples
        self.context_prompt_idx = context_prompt_idx
        self.cache_prompt_idx = cache_prompt_idx
        self.instruction_prompt = self.get_instruction_prompt()
        self.adjacent_words = self.get_adjacent_words()
        self.boilerplate_words = self.get_boilerplate_words()
        self.varied_instructions = self.get_varied_instructions()
        self.varied_labels = self.get_varied_label_space()

    def load_cache_prompts(self):
        prompts_for_cache = ["",  # Cache gets instruction only
                             ]
        cache_prompts = [self.instruction_prompt + prompt for prompt in prompts_for_cache]
        if self.cache_prompt_idx is not None:
            cache_prompts = [cache_prompts[self.cache_prompt_idx]]
        return cache_prompts
    
    def get_boilerplate_words(self):
        return ["Speaker", "The", "What", "Sure", "Options", "Please", "1"]
    
    def get_adjacent_words(self):
        return []
        
    def get_instruction_prompt(self):
        instruction_prompts = {
            "anim_color": 
                "Given an animal, state its most typical associated color.",
            "can_fly": 
                "Given an animal, state whether or not it can fly. Print 'yes' or 'no'.",
            "m_color": 
                "Given a mushroom, state its most typical associated color.",
            "m_edible":
                "Given a mushroom, state whether or not it is edible. Print 'yes' or 'no'.",
            "adj_comp":
                "Given an adjective, state its comparative form.",
            "adj_ant":
                "Given an adjective, state its antonym.",
            "repeat_last":
                "Given a phrase, simply repeat the last word of the phrase.",
            "adv_ans":
                "Given two operands, simply print 'purple'.",
            "basic_ans":
                "Given two operands, perform addition.",
            }
        return instruction_prompts[self.subtask]
    
    def get_varied_instructions(self):
        data_dir = os.path.join(os.path.dirname(__file__), "local_tasks", "varied_instructions", f"{self.subtask}_rephrased.csv")
        df = pd.read_csv(data_dir)
        d = Dataset.from_pandas(df)
        return d["instruction"]
    
    def get_varied_label_space(self):
        varied_label_space = {
            "anim_color": [],
            "can_fly": [],
            "m_color": [],
            "m_edible": [],
            "adj_comp": [],
            "adj_ant": [],
            "repeat_last": [],
            "adv_ans": [],
            "basic_ans": [],
            }
        return varied_label_space[self.subtask]
    
    def _format_with_instruction(self, sample):
        query_to_field = {
            "animals": "Animal",
            "objects": "Object",
            "mushrooms": "Mushroom",
            "adjectives": "Adjective",
            "en_words": "Phrase",
            "math": "Operands",
        }
        subtask_to_options = {
            "anim_color": [
                "red", 
                "orange", 
                "yellow", 
                "golden", 
                "green",
                "blue", 
                "purple",
                "pink",
                "brown",
                "black",
                "white",
                "gray",
                "silver",
                "tan"
                ],
            "can_fly": ["yes", "no"],
            "m_color": [
                "red", 
                "orange", 
                "yellow", 
                "golden", 
                "green",
                "blue", 
                "purple",
                "pink",
                "brown",
                "black",
                "white",
                "gray",
                "silver",
                "tan"
                ],
            "m_edible": ["yes", "no"],
            "adj_comp": [],
            "adj_ant": [],
            "repeat_last": [],
            "basic_ans": [],
            "adv_ans": [],
        }
        if self.subtask == "repeat_last":
            sample["query"] = sample["trans_instr"] + " " + sample['query']
            sample["instr+target"] = f"{self.instruction_prompt} {query_to_field[self.base_query]}: {sample['query']}."
        else:
            sample["instr+target"] = f"{self.instruction_prompt} {query_to_field[self.base_query]}: {sample['query']}. Answer:"
        sample["mod_options"] = subtask_to_options[self.subtask]
        sample["mod_label"] = sample[self.subtask]
        sample["prompt"] = f"{query_to_field[self.base_query]}: {sample['query']}. Answer:"
        sample["instr"] = self.instruction_prompt

        # Cast numerical tasks to string
        if self.subtask == "basic_ans":
            sample["mod_label"] = str(sample["mod_label"])
            sample["basic_ans"] = str(sample["basic_ans"])
        return sample

    def load_test(self):
        data_dir = os.path.join(os.path.dirname(__file__), "local_tasks", f"{self.base_query}.csv")
        df = pd.read_csv(data_dir)
        d = Dataset.from_pandas(df)
        d = d.map(self._format_with_instruction)
        if self.num_samples is not None:
            d = d.select(range(0, self.num_samples))
        return d


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.instr = self.data["instr"]
        self.inputs = self.data["instr+target"]
        self.targets = self.data["mod_label"]
        self.options = self.data["mod_options"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return [self.inputs[index], self.targets[index], self.options[index], self.instr[index]]
    

def load_task(given_name=None, subtask=None, tokenizer=None):
    if given_name is not None:
        task_name = given_name
    else:
        task_name = config.args.task

    # Load task instructions
    if task_name in [
        "implicatures", 
        "irony_identification", 
        "metaphor_boolean",
        "nonsense_words_grammar",
        "object_counting",
        "snarks",
        "strange_stories"
        ]:
        task = BigBenchTask(
            task_name,
            num_samples = config.args.num_samples,
            tokenizer=tokenizer,
            cache_prompt_idx=config.args.cache_prompt_idx
        )
    elif task_name in [
        "animals",
        "objects",
        "mushrooms",
        "adjectives",
        "en_words",
        "math"
        ]:
        task = CsvTask(
            task_name,
            subtask,
            num_samples = config.args.num_samples,
            cache_prompt_idx=config.args.cache_prompt_idx
        )
    else:
        raise Exception("Unknown Task:", task_name)
    return task
