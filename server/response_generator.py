import torch
from nltk import tokenize
from torch.nn.functional import softmax

from server.history import History
from server.sentiment import Sentiment
from utils.helper import load_classifier, EOS_ID
from utils.utils_sample import scorer


class ResponseGenerator:
    """
    Class used to generate sentiment colored responses given an input.
    The code in this class was adapted from interact_adapter.py
    """

    def __init__(self, args, model, tokenizer):
        self._args = args
        self._model = model
        self._tokenizer = tokenizer
        self._device = 'cuda'  # this can be changed in the future to allow for multiple kinds of devices
        self.__history = History(args.max_history)
        self.__classifiers = self.__get_classifiers(self._args, self._model)

    @staticmethod
    def __get_classifiers(args, model):
        classifiers = {}

        args.discrim = "sentiment"
        args.label_class = Sentiment.POSITIVE.get_label_class()
        classifier, class2idx = load_classifier(args, model)
        classifiers[Sentiment.POSITIVE] = [classifier, class2idx]

        args.discrim = "sentiment"
        args.label_class = Sentiment.NEGATIVE.get_label_class()
        classifier, class2idx = load_classifier(args, model)
        classifiers[Sentiment.NEGATIVE] = [classifier, class2idx]

        return classifiers

    @staticmethod
    def __top_k_logits(logits, k, probs=False):
        """
        Masks everything but the k top entries as -infinity (1e10).
        Used to mask logits such that e^-infinity -> 0 won't contribute to the sum of the denominator.
        """

        if k == 0:
            return logits

        values = torch.topk(logits, k)[0]
        batch_minimums = values[:, -1].view(-1, 1).expand_as(logits)

        # don't entirely understand why this is working, as I think the parameters should be mismatched
        # TODO: Look into why this is working
        if probs:
            return torch.where(logits < batch_minimums, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_minimums, torch.ones_like(logits) * -1e10, logits)

    def __sample(self, context, sample=True):
        output = torch.tensor(context, device=self._device, dtype=torch.long)
        output_response = output.new_zeros([output.size(0), 0])
        stopped = [0 for _ in range(output.size(0))]
        for i in range(self._args.repetition_penalty.length):
            prev = output[:, -1:]
            _, past = self._model(output[:, :-1])

            logits, past = self._model(prev, past=past)
            logits = logits[:, -1, :] / self._args.repetition_penalty.temperature  # + SmallConst

            for output_element_idx, output_element in enumerate(output):
                for token_idx in set(output_element.tolist()):
                    if logits[output_element_idx, token_idx] < 0:
                        logits[output_element_idx, token_idx] *= self._args.repetition_penalty
                    else:
                        logits[output_element_idx, token_idx] /= self._args.repetition_penalty

            logits = self.__top_k_logits(logits, k=self._args.top_k)  # + SmallConst

            log_probs = softmax(logits, dim=-1)

            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)

            output = torch.cat((output, prev), dim=1)
            output_response = torch.cat((output_response, prev), dim=1)

            for prev_element_idx, prev_element in enumerate(prev.tolist()):
                if (prev_element[0]) == EOS_ID:
                    stopped[prev_element_idx] = 1

            if all(x == 1 for x in stopped):
                break

        return output_response

    def get_response(self, sentiment: Sentiment, utterance: str) -> str:
        classifier, class2idx = self.__classifiers[sentiment]
        self._args.num_samples = 10
        self._args.label_class = sentiment.get_label_class()

        self.__history.append(utterance)

        context_tokens = sum([self._tokenizer.encode(h) + [EOS_ID] for h in self.__history], [])
        context_tokens = [context_tokens for _ in range(self._args.num_samples)]

        original_sentence = self.__sample(context=context_tokens)
        spk_turn = {"text": original_sentence.tolist()}
        hypothesis, _, _ = scorer(self._args, spk_turn, classifier, self._tokenizer, class2idx,
                                  knowledge=None, plot=False)
        response = hypothesis[0][-1]
        response = " ".join(tokenize.sent_tokenize(response)[:2])

        self.__history.append(response)

        return response
