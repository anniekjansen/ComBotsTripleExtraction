import sys
sys.path.append('predicate_normalization')

from src.model_transformer.argument_extraction import ArgumentExtraction
from src.model_transformer.triple_scoring import TripleScoring
from src.model_transformer.post_processing import PostProcessor
from src.model_transformer.utils import pronoun_to_speaker_id, speaker_id_to_speaker, bio_tags_to_tokens

from itertools import product
import spacy


class AlbertTripleExtractor:
    def __init__(self, path, bio_lookup, base_model='albert-base-v2',
                 sep='<eos>', speaker1='speaker1', speaker2='speaker2'):
        """ Constructor of the Albert-based Triple Extraction Pipeline.

        :param path:       path to savefile
        :param base_model: base model (default: albert-base-v2)
        :param sep:        separator token used to delimit dialogue turns (default: <eos>)
        :param speaker1:   name of user (default: speaker1)
        :param speaker2:   name of system (default: speaker2)
        """
        self._argument_module = ArgumentExtraction(base_model, path=path)
        self._scoring_module = TripleScoring(base_model, path=path)

        self._post_processor = PostProcessor()
        self._nlp = spacy.load('en_core_web_sm')
        self._sep = sep

        # Load lookup for bio annotations for predicates
        self._bio_lookup = bio_lookup

        # Assign identities to speakers
        self._speaker1 = speaker1
        self._speaker2 = speaker2

    @property
    def name(self):
        return "ALBERT"

    def _tokenize(self, dialog):
        """ Divides up the dialogue into separate turns and dereferences
            personal pronouns 'I' and 'you'.

        :param dialog: separator-delimited dialogue
        :return:       list of tokenized dialogue turns
        """
        # Split dialogue into turns
        turns = [turn.lower().strip() for turn in dialog.split(self._sep)]

        # Tokenize each turn separately (and splitting "n't" off)
        tokens = []
        for turn_id, turn in enumerate(turns):
            # Assign speaker ID to turns (tn=1, tn-1=0, tn-2=1, etc.)
            speaker_id = (len(turns) - turn_id + 1) % 2
            if turn:
                tokens += [pronoun_to_speaker_id(t.lower_, speaker_id) for t in self._nlp(turn)] + ['<eos>']
        return tokens

    def extract_triples(self, dialog, post_process=True, batch_size=32, verbose=True, ):
        """

        :param dialog:       separator-delimited dialogue
        :param post_process: Whether to apply rules to fix contractions and strip auxiliaries (like baselines)
        :param batch_size:   If a lot of possible triples exist, batch up processing
        :param verbose:      whether to print messages (True) or be silent (False) (default: True)
        :return:             A list of confidence-triple pairs of the form (confidence, (subj, pred, obj, polarity))
        """
        # Assign unambiguous tokens to you/I
        tokens = self._tokenize(dialog)

        # Extract SPO arguments from token sequence
        subjs, preds, objs, subwords = self._argument_module.predict(tokens)

        # Decode predictions into strings
        subj_args = bio_tags_to_tokens(subwords, subjs.T, self._bio_lookup, one_hot=True)
        # pred_args = bio_tags_to_tokens(subwords, preds.T, self._bio_lookup, one_hot=True) # change predicate to True
        pred_args = bio_tags_to_tokens(subwords, preds.T, self._bio_lookup, predicate=True, one_hot=True)  
        obj_args = bio_tags_to_tokens(subwords, objs.T, self._bio_lookup, one_hot=True)

        
        if verbose:
            print('subjects:   %s' % subj_args)
            print('predicates: %s' % pred_args)
            print('objects:    %s\n' % obj_args)

        # List all possible combinations of arguments
        candidates = [list(triple) for triple in product(subj_args, pred_args, obj_args)]
        if not candidates:
            return []

        # Score candidate triples
        predictions = []
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            for y_hat in self._scoring_module.predict(tokens, batch):
                predictions.append(y_hat)

        # Rank candidates according to entailment predictions
        triples = []
        for y_hat, (subj, pred, obj) in zip(predictions, candidates):
            pol = 'negative' if y_hat[2] > y_hat[1] else 'positive'
            ent = max(y_hat[1], y_hat[2])

            # Replace SPEAKER* with speaker
            subj = speaker_id_to_speaker(subj, self._speaker1, self._speaker2)
            pred = speaker_id_to_speaker(pred, self._speaker1, self._speaker2)
            obj = speaker_id_to_speaker(obj, self._speaker1, self._speaker2)

            # Fix mistakes, expand contractions
            if post_process:
                subj, pred, obj = self._post_processor.format((subj, pred, obj))

            triples.append((ent, (subj, pred, obj, pol)))

        return sorted(triples, key=lambda x: -x[0])


if __name__ == '__main__':
    # bio_lookup = {3: 'like', 5: 'do'}
    bio_lookup_l1 = {3: 'act', 5: 'add', 7: 'afford', 9: 'agree', 11: 'aid', 13: 'allow', 15: 'anticipate', 17: 'arrest', 19: 'arrive', 21: 'ask', 23: 'be', 25: 'be angry', 27: 'be arrested', 29: 'be at', 31: 'be available', 33: 'be certain', 35: 'be difficult', 37: 'be free', 39: 'be from', 41: 'be happy', 43: 'be in', 45: 'be in between', 47: 'be old', 49: 'be on', 51: 'be out', 53: 'be over', 55: 'be ready', 57: 'be scared', 59: 'be with', 61: 'be wrong', 63: 'become', 65: 'believe', 67: 'betray', 69: 'better than', 71: 'borrow', 73: 'break', 75: 'break up', 77: 'bring', 79: 'burn', 81: 'buy', 83: 'call', 85: 'can', 87: 'cancel', 89: 'care', 91: 'catch', 93: 'cause', 95: 'change', 97: 'check in', 99: 'choose', 101: 'clean', 103: 'close', 105: 'come', 107: 'confuse', 109: 'consider', 111: 'contact', 113: 'cook', 115: 'copy', 117: 'cost', 119: 'could', 121: 'count', 123: 'cut', 125: 'dance', 127: 'decide', 129: 'depend on', 131: 'destroy', 133: 'devote', 135: 'die', 137: 'different', 139: 'dislike', 141: 'distrust', 143: 'do', 145: 'do badly', 147: 'do well', 149: 'draw', 151: 'dress', 153: 'drink', 155: 'drive', 157: 'drop', 159: 'earn', 161: 'eat', 163: 'enable', 165: 'endure', 167: 'equal', 169: 'exchange', 171: 'exercise', 173: 'expect', 175: 'fall', 177: 'feel', 179: 'fight', 181: 'fill', 183: 'find', 185: 'finish', 187: 'fit', 189: 'fix', 191: 'follow', 193: 'forget', 195: 'gain', 197: 'get', 199: 'get in', 201: 'give', 203: 'go', 205: 'grow', 207: 'handle', 209: 'harder than', 211: 'have', 213: 'hear', 215: 'help', 217: 'hit', 219: 'hold', 221: 'hope', 223: 'hurry', 225: 'hurt', 227: 'include', 229: 'investigate', 231: 'involve', 233: 'join', 235: 'joke', 237: 'keep', 239: 'keep secret', 241: 'kill', 243: 'know', 245: 'lead', 247: 'learn', 249: 'leave', 251: 'light', 253: 'like', 255: 'limit', 257: 'listen', 259: 'live', 261: 'live in', 263: 'located', 265: 'lock out', 267: 'look', 269: 'lose', 271: 'love', 273: 'made of', 275: 'make', 277: 'marry', 279: 'match', 281: 'mean', 283: 'meet', 285: 'miss', 287: 'misspell', 289: 'mix', 291: 'more expensive than', 293: 'motivate', 295: 'move', 297: 'must', 299: 'need', 301: 'None', 303: 'occupy', 305: 'open', 307: 'order', 309: 'owe', 311: 'own', 313: 'paint', 315: 'park', 317: 'pay', 319: 'pay attention', 321: 'pick', 323: 'plan', 325: 'play', 327: 'portray', 329: 'practice', 331: 'pray', 333: 'prefer', 335: 'prepare', 337: 'pretend', 339: 'prevent', 341: 'promise', 343: 'propose', 345: 'protect', 347: 'pull', 349: 'put', 351: 'rain', 353: 'raise', 355: 'reach', 357: 'read', 359: 'recognize', 361: 'recommend', 363: 'recycle', 365: 'relationship', 367: 'relax', 369: 'remain', 371: 'remember', 373: 'remove', 375: 'reserve', 377: 'respond', 379: 'retire', 381: 'return', 383: 'rule', 385: 'run', 387: 'save', 389: 'say', 391: 'see', 393: 'seek', 395: 'seem', 397: 'sell', 399: 'send', 401: 'share', 403: 'shoot', 405: 'should', 407: 'show', 409: 'sign', 411: 'similar', 413: 'sing', 415: 'sit', 417: 'sleep', 419: 'smell', 421: 'smoke', 423: 'socialize', 425: 'sort', 427: 'speak', 429: 'spend', 431: 'start', 433: 'starve', 435: 'stay', 437: 'steal', 439: 'stop', 441: 'study', 443: 'survive', 445: 'swim', 447: 'take', 449: 'take off', 451: 'take out', 453: 'talk', 455: 'taste', 457: 'teach', 459: 'tell', 461: 'think', 463: 'throw', 465: 'tired', 467: 'travel', 469: 'try', 471: 'use', 473: 'visit', 475: 'wait', 477: 'wake up', 479: 'walk', 481: 'want', 483: 'waste', 485: 'watch', 487: 'water', 489: 'wear', 491: 'will', 493: 'win', 495: 'work', 497: 'work with', 499: 'worry', 501: 'would', 503: 'write'}
    
    # model = AlbertTripleExtractor('models/2022-04-27', bio_lookup)
    model = AlbertTripleExtractor('models/level1', bio_lookup_l1)


    # Test!
    example = "I went to the new university. It was great! <eos> I like studying too and learning. You? <eos> No, hate it!"

    for score, triple in model.extract_triples(example):
        print(score, triple)
