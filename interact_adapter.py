from tabulate import tabulate
from utils.helper import load_classifier
from utils.helper import EOS_ID
from utils.utils_sample import scorer
import torch.nn.functional as F
import torch
from nltk import tokenize

tabulate.PRESERVE_WHITESPACE = True


def top_k_logits(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def sample(model, args, context=None, past=None, device='cuda',
           sample=True, repetition_penalty=1.0):
    output = torch.tensor(context, device=device, dtype=torch.long) if context else None
    output_response = output.new_zeros([output.size(0), 0])
    stopped = [0 for _ in range(output.size(0))]
    for i in range(args.length):

        if past is None and output is not None:
            prev = output[:, -1:]
            _, past = model(output[:, :-1])

        logits, past = model(prev, past=past)

        logits = logits[:, -1, :] / args.temperature  # + SmallConst
        for i_o, o_ in enumerate(output):
            for token_idx in set(o_.tolist()):
                if logits[i_o, token_idx] < 0:
                    logits[i_o, token_idx] *= repetition_penalty
                else:
                    logits[i_o, token_idx] /= repetition_penalty

        logits = top_k_logits(logits, k=args.top_k)  # + SmallConst
        log_probs = F.softmax(logits, dim=-1)

        if sample:
            prev = torch.multinomial(log_probs, num_samples=1)
        else:
            _, prev = torch.topk(log_probs, k=1, dim=-1)

        output = prev if output is None else torch.cat((output, prev), dim=1)  # update output
        output_response = torch.cat((output_response, prev), dim=1)

        for i_p, p in enumerate(prev.tolist()):
            if (p[0]) == EOS_ID:
                stopped[i_p] = 1

        if all(x == 1 for x in stopped):
            break

    return output_response


def get_rankers(args, model):
    classifiers = {}

    args.discrim = 'sentiment'
    args.label_class = 2
    classifier, class2idx = load_classifier(args, model)
    classifiers['a'] = [classifier, class2idx]

    args.discrim = 'sentiment'
    args.label_class = 3
    classifier, class2idx = load_classifier(args, model)
    classifiers['b'] = [classifier, class2idx]

    return classifiers


def interact(args, model, enc, classifier, class2idx, device):
    classifiers = get_rankers(args, model)
    history = []
    while True:
        raw_text = input("USR >>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("USR >>>")

        style = input(
            "Choose a style \n (a) Positive (b) Negative \n >>> ")
        if style == "a":
            classifier, class2idx = classifiers["a"]
            args.num_samples = 10
            task_id = 1
            args.label_class = 2
        elif style == "b":
            classifier, class2idx = classifiers["b"]
            args.num_samples = 10
            task_id = 0
            args.label_class = 3
        else:
            args.num_samples = 1
            args.label_class = 0
            task_id = -1

        history.append(raw_text)

        context_tokens = sum([enc.encode(h) + [EOS_ID] for h in history], [])
        context_tokens = [context_tokens for _ in range(args.num_samples)]

        original_sentence = sample(model=model, args=args, context=context_tokens, device=device,
                                   repetition_penalty=args.repetition_penalty)
        spk_turn = {"text": original_sentence.tolist()}
        hypothesis, _, _ = scorer(args, spk_turn, classifier, enc, class2idx, knowledge=None, plot=False)
        text = hypothesis[0][-1]
        text = " ".join(tokenize.sent_tokenize(text)[:2])

        print(f"SYS >>> {text}")
        history.append(text)
        history = history[-(2 * args.max_history + 1):]
