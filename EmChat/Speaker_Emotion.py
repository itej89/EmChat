# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
import torch.nn.functional as F
from config import InteractConfig

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as nnf
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from config import Config
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from ignite.contrib.handlers.tensorboard_logger import *

from pytorch_pretrained_bert import (OpenAIAdam, OpenAIGPTDoubleHeadLMEmotionRecognitionModel, OpenAIGPTTokenizer,
                                     GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME,
                                     BertModel, BertTokenizer)

from utils import get_dataset, get_dataset_for_daily_dialog

import numpy as np

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>",
                  "<no_emotion>", "<happiness>", "<surprise>", "<sadness>", "<disgust>", "<anger>", "<fear>",
                  "<directive>", "<inform>", "<commissive>", "<question>",
                  "<pad>"]
MODEL_INPUTS  = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids", "token_emotion_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids", "token_emotion_ids"]

logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, config):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if config.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=config.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def get_emotion_label(tokenizer, candidate_emotion):
    _, _, _, _, no_emotion_id, happiness_id, surprise_id, sadness_id, disgust_id, anger_id, fear_id, _, _, _, _, _ = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if candidate_emotion == happiness_id:
        return 0
    elif candidate_emotion == surprise_id:
        return 1
    elif candidate_emotion == sadness_id:
        return 2
    elif candidate_emotion == disgust_id:
        return 3
    elif candidate_emotion == anger_id:
        return 4
    elif candidate_emotion == fear_id:
        return 5
    elif candidate_emotion == no_emotion_id:
        return 6


def build_input_from_segments(history, emotions, reply, tokenizer, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:4])
    #tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])

    instance = {}
    # sequence = [[bos] + history[0] + list(chain(*history[1:]))]  + [reply + ([eos] if with_eos else [])] #seq = [personas, history, reply] concatenate all persona sentences
    sequence = [[bos] + history[0]] + history[1:] + [reply + ([eos] if with_eos else [])]
    sequence = [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence)]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s] # the last for is for repeating the speaker1 and speaker2 for all tokens
    #instance["token_emotion_ids"] = [emotions[i] for i, s in enumerate(sequence[:-1]) for _ in s] + [true_emotion] * len(sequence[-1])
    instance["token_emotion_ids"] = [emotions[i] for i, s in enumerate(sequence[:-1]) for _ in range(len(s)+1)]

    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    # instance["mc_labels"] = get_emotion_label(tokenizer, true_emotion)
    # instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:] #all -1 except for reply, reply is just the ids
    return instance, sequence



def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

EMOTIONS_TOKEN_ID = {"NO_EMOTION":40482, "HAPPINESS":40483, "SURPRISE":40484, "SADNESS":40485, "DISGUST":40486, "ANGRY":40487, "FEAR":40488}
def sample_sequence(history, tokenizer, model, args, SPECIAL_TOKENS, reply=None, emotion_history=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    if reply is None:
        reply = []

    candidate_emotion = []
    candidate_act = []


    for i in range(args.max_length):
        topic = 40492
        emotions = emotion_history
        actions = [40499]*len(history)
        instance, sequence = build_input_from_segments(history, emotions, reply, tokenizer,
                                                       with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        mc_token_ids = torch.tensor(instance["mc_token_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        token_emotion_ids = torch.tensor(instance["token_emotion_ids"], device=args.device).unsqueeze(0)

        lm_logits, mc_logits  = model(input_ids, mc_token_ids, token_type_ids=token_type_ids, token_emotion_ids=token_emotion_ids)

        emotion_indices = torch.argmax(mc_logits, dim=1)
        
        # logits = logits.squeeze(0)
        # if "gpt2" == args.model:
        #     logits = logits[0]
        # logits = logits[0, -1, :] / args.temperature
        # logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        # probs = F.softmax(logits, dim=-1)

        # prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        # if i < args.min_length and prev.item() in special_tokens_ids:
        #     while prev.item() in special_tokens_ids:
        #         prev = torch.multinomial(probs, num_samples=1)

        # if prev.item() in special_tokens_ids:
        #     break
        # reply.append(prev.item())

        # candidate_emotion.append(EMOTIONS[emotion_indices[0]])
    prob = nnf.softmax(mc_logits, dim=1)
    return prob

config = None
tokenizer = None
model = None
out_ids = None
candidate_emotion = None
emotion_history = []
history = [] 



EMOTIONS = ["Joy", "Surprise", "Sadness", "Disgust", "Anger", "Fear"]

def Load_Model():
    global model, tokenizer, config
    config_file = "configs/interact_emotion_config.json"
    config = InteractConfig.from_json_file(config_file)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(config))

    if config.model_checkpoint == "":
        config.model_checkpoint = download_pretrained_model()

    torch.random.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    logger.info("Get pretrained model and tokenizer")
    if config.model == "bert":
        tokenizer_class = BertTokenizer
        model_class = BertLMHeadModel
    elif config.model == "gpt2":
        tokenizer_class = GPT2Tokenizer
        model_class = GPT2LMHeadModel
    else:
        tokenizer_class = OpenAIGPTTokenizer
        model_class = OpenAIGPTDoubleHeadLMEmotionRecognitionModel


    tokenizer = tokenizer_class.from_pretrained(config.model_checkpoint)
    model = model_class.from_pretrained(config.model_checkpoint)

    model.to(config.device)
    model.eval()

def predict_emotion(sentence):
    global model, tokenizer, out_ids, candidate_emotion, emotion_history, history

    history.append(tokenizer.encode(sentence))
    emotion_history.append(EMOTIONS_TOKEN_ID["HAPPINESS"])
    with torch.no_grad():
        candidate_emotion = sample_sequence(history, tokenizer, model, config, SPECIAL_TOKENS, None, emotion_history)
    # history = history[-(2 * config.max_history + 1):]
    # emotion_history = emotion_history[-(2 * config.max_history + 1):]
    history.clear()
    emotion_history.clear()
    # out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    # print(f"{candidate_emotion[len(candidate_emotion)-1]}")

    
    total = torch.sum(candidate_emotion[0]).item()
    Joy_per = (candidate_emotion[0][0].item()/total)*100
    Surprise_per = (candidate_emotion[0][1].item()/total)*100
    Sadness_per = (candidate_emotion[0][2].item()/total)*100
    Disgust_per = (candidate_emotion[0][3].item()/total)*100
    Anger_per = (candidate_emotion[0][4].item()/total)*100
    Fear_per = (candidate_emotion[0][5].item()/total)*100

    total = Joy_per + Surprise_per + Sadness_per + Disgust_per + Anger_per + Fear_per

    return ({'ANGER':str(Anger_per), 'DISGUST':str(Disgust_per), 'FEAR':str(Fear_per), 'JOY':str(Joy_per), 'SADNESS':str(Sadness_per), 'SURPRISE':str(Surprise_per)})