import sys
sys.path.append('src/model_transformer')

from src.model_transformer.run_transformer_pipeline import AlbertTripleExtractor
from src.evaluation.benchmark_evaluation.metrics import classification_report
from pathlib import Path
import json
import spacy




def load_examples(path):
    """ Load examples in the form of (str: dialogue, list: triples).

    :param path: Path to test file (e.g. 'test_examples/test_full.txt')
    :return:     List of (str: dialogue, list: triples) pairs.
    """
    # Extract lines corresponding to dialogs and triples
    samples = []
    with open(path, 'r', encoding='utf-8') as file:
        block = []
        for line in file:
            if line.strip() and not line.startswith('#'):
                block.append(line.strip())
            elif block:
                samples.append(block)
                block = []
        samples.append(block)

    # Split triple arguments
    examples = []
    for block in samples:
        dialog = block[1]
        triples = [string_to_triple(triple) for triple in block[2:]]
        examples.append((dialog, triples))

    return examples


def save_results(result, model, test_file, confidence):
    with open('src/results/results/' + Path(test_file).stem + '.json', 'w') as outfile:
        dct = {'precision': result[0], 'recall': result[1], 'f1': result[2], 'auc': result[3], 'pr-curve': result[4]}
        json.dump(dct, outfile)

def string_to_triple(text_triple):
    """ Tokenizes triple line into individual arguments

    :param text_triple: plain-text string of triple
    :return:            triple of the form (subj, pred, obj, polar)
    """
    return tuple([x.strip() for x in text_triple.split(',')])


def lemmatize_triple(subj, pred, obj, polar, nlp):
    """ Takes in a triple and perspective and lemmatizes/normalizes the predicate.
    """
    pred = ' '.join([t.lemma_ for t in nlp(pred)])
    return subj, pred, obj, polar


def evaluate(test_file, model, num_samples=-1, k=0.9, deduplication=True):
    """ Evaluates the model on a test file, yielding scores for precision@k,
        recall@k, F1@k and PR-AUC.

    :param test_file:     Test file from '/test_examples'
    :param model:         Albert, Dependency or baseline model instance
    :param num_samples:   The maximum number of samples to evaluate (default: all)
    :param k:             Confidence level at which to evaluate models
    :param deduplication: Whether to lemmatize predicates to make sure duplicate predicates such as "is"
                          and "are" are removed and match across baselines (default: True)
    :return:              Scores for precision@k, recall@k, F1@k and PR-AUC
    """
    # Extract dialog-triples pairs from annotations
    examples = load_examples(test_file)
    if num_samples > 0:
        examples = examples[:num_samples]

    # Predictions
    true_triples = []
    pred_triples = []
    for i, (dialog, triples) in enumerate(examples):
        # Print progress
        print('\n (%s/%s) input: %s' % (i + 1, len(examples), dialog))

        # Predict triples
        extractions = model.extract_triples(dialog,  verbose=True)

        # Check for error in test set formatting
        error = False
        for triple in triples:
            if len(triple) != 4:
                print('#######\nERROR\n#######')
                print(triple)
                error = True

        print('expected:', triples)
        print('found:   ', [t for c, t in extractions if c > k])

        # save expected and found to file:
        with open("src/results/results/" + Path(test_file).stem + ".txt", 'a', encoding='utf-8') as outf:
            outf.write(str('\n(%s/%s) input: %s' % (i + 1, len(examples), dialog)))
            outf.write('\nextracted predicates:'+str([t[1][1] for t in extractions]))
            outf.write('\nexpected: '+str(triples))
            outf.write('\nfound:   '+str([t for c, t in extractions if c > k])+'\n')

        if not error:
            true_triples.append(triples)
            pred_triples.append(extractions)

    # If lemmatize is enabled, map word forms to lemmas
    if deduplication:
        print('\nPerforming de-duplication')
        nlp = spacy.load('en_core_web_sm')
        true_triples = [set([lemmatize_triple(*triple, nlp) for triple in lst]) for lst in true_triples]
        pred_triples = [set([(conf, lemmatize_triple(*triple, nlp)) for conf, triple in lst]) for lst in pred_triples]

    # Compute performance metrics
    return classification_report(true_triples, pred_triples, k=k)


if __name__ == '__main__':
    MODEL = 'albert'
    # TEST_FILE = '../../dataset/final/test/test_declarative_statements.txt'
    #TEST_FILE = Path("src/dataset/final/test/test_declarative_statements.txt")

    TEST_FILE = Path("src/dataset/final/eval/test_declarative_statements_level1_eval.txt")
    # TEST_FILE = Path("src/dataset/final/eval/test_coreference_level1_eval.txt")
    # TEST_FILE = Path("src/dataset/final/eval/test_single_utterances_level1_eval.txt")

    # TEST_FILE = Path("src/dataset/final/eval/test_declarative_statements_level2_eval.txt")
    # TEST_FILE = Path("src/dataset/final/eval/test_coreference_level2_eval.txt")
    # TEST_FILE = Path("src/dataset/final/eval/test_single_utterances_level2_eval.txt")

    MIN_CONF = 0.9

    bio_lookup_l1 = {3: 'act', 5: 'add', 7: 'afford', 9: 'agree', 11: 'aid', 13: 'allow', 15: 'anticipate', 17: 'arrest', 19: 'arrive', 21: 'ask', 23: 'be', 25: 'be angry', 27: 'be arrested', 29: 'be at', 31: 'be available', 33: 'be certain', 35: 'be difficult', 37: 'be free', 39: 'be from', 41: 'be happy', 43: 'be in', 45: 'be in between', 47: 'be old', 49: 'be on', 51: 'be out', 53: 'be over', 55: 'be ready', 57: 'be scared', 59: 'be with', 61: 'be wrong', 63: 'become', 65: 'believe', 67: 'betray', 69: 'better than', 71: 'borrow', 73: 'break', 75: 'break up', 77: 'bring', 79: 'burn', 81: 'buy', 83: 'call', 85: 'can', 87: 'cancel', 89: 'care', 91: 'catch', 93: 'cause', 95: 'change', 97: 'check in', 99: 'choose', 101: 'clean', 103: 'close', 105: 'come', 107: 'confuse', 109: 'consider', 111: 'contact', 113: 'cook', 115: 'copy', 117: 'cost', 119: 'could', 121: 'count', 123: 'cut', 125: 'dance', 127: 'decide', 129: 'depend on', 131: 'destroy', 133: 'devote', 135: 'die', 137: 'different', 139: 'dislike', 141: 'distrust', 143: 'do', 145: 'do badly', 147: 'do well', 149: 'draw', 151: 'dress', 153: 'drink', 155: 'drive', 157: 'drop', 159: 'earn', 161: 'eat', 163: 'enable', 165: 'endure', 167: 'equal', 169: 'exchange', 171: 'exercise', 173: 'expect', 175: 'fall', 177: 'feel', 179: 'fight', 181: 'fill', 183: 'find', 185: 'finish', 187: 'fit', 189: 'fix', 191: 'follow', 193: 'forget', 195: 'gain', 197: 'get', 199: 'get in', 201: 'give', 203: 'go', 205: 'grow', 207: 'handle', 209: 'harder than', 211: 'have', 213: 'hear', 215: 'help', 217: 'hit', 219: 'hold', 221: 'hope', 223: 'hurry', 225: 'hurt', 227: 'include', 229: 'investigate', 231: 'involve', 233: 'join', 235: 'joke', 237: 'keep', 239: 'keep secret', 241: 'kill', 243: 'know', 245: 'lead', 247: 'learn', 249: 'leave', 251: 'light', 253: 'like', 255: 'limit', 257: 'listen', 259: 'live', 261: 'live in', 263: 'located', 265: 'lock out', 267: 'look', 269: 'lose', 271: 'love', 273: 'made of', 275: 'make', 277: 'marry', 279: 'match', 281: 'mean', 283: 'meet', 285: 'miss', 287: 'misspell', 289: 'mix', 291: 'more expensive than', 293: 'motivate', 295: 'move', 297: 'must', 299: 'need', 301: 'None', 303: 'occupy', 305: 'open', 307: 'order', 309: 'owe', 311: 'own', 313: 'paint', 315: 'park', 317: 'pay', 319: 'pay attention', 321: 'pick', 323: 'plan', 325: 'play', 327: 'portray', 329: 'practice', 331: 'pray', 333: 'prefer', 335: 'prepare', 337: 'pretend', 339: 'prevent', 341: 'promise', 343: 'propose', 345: 'protect', 347: 'pull', 349: 'put', 351: 'rain', 353: 'raise', 355: 'reach', 357: 'read', 359: 'recognize', 361: 'recommend', 363: 'recycle', 365: 'relationship', 367: 'relax', 369: 'remain', 371: 'remember', 373: 'remove', 375: 'reserve', 377: 'respond', 379: 'retire', 381: 'return', 383: 'rule', 385: 'run', 387: 'save', 389: 'say', 391: 'see', 393: 'seek', 395: 'seem', 397: 'sell', 399: 'send', 401: 'share', 403: 'shoot', 405: 'should', 407: 'show', 409: 'sign', 411: 'similar', 413: 'sing', 415: 'sit', 417: 'sleep', 419: 'smell', 421: 'smoke', 423: 'socialize', 425: 'sort', 427: 'speak', 429: 'spend', 431: 'start', 433: 'starve', 435: 'stay', 437: 'steal', 439: 'stop', 441: 'study', 443: 'survive', 445: 'swim', 447: 'take', 449: 'take off', 451: 'take out', 453: 'talk', 455: 'taste', 457: 'teach', 459: 'tell', 461: 'think', 463: 'throw', 465: 'tired', 467: 'travel', 469: 'try', 471: 'use', 473: 'visit', 475: 'wait', 477: 'wake up', 479: 'walk', 481: 'want', 483: 'waste', 485: 'watch', 487: 'water', 489: 'wear', 491: 'will', 493: 'win', 495: 'work', 497: 'work with', 499: 'worry', 501: 'would', 503: 'write'}
    bio_lookup_l2 = {3: 'act', 5: 'add', 7: 'agree', 9: 'ask', 11: 'aux', 13: 'be', 15: 'be difficult', 17: 'be free', 19: 'be from', 21: 'be old', 23: 'be ready', 25: 'be wrong', 27: 'become', 29: 'betray', 31: 'better than', 33: 'borrow', 35: 'bring', 37: 'call', 39: 'cause', 41: 'change', 43: 'check in', 45: 'choose', 47: 'clean', 49: 'close', 51: 'come', 53: 'commerce-transaction', 55: 'confuse', 57: 'consume', 59: 'contact', 61: 'cook', 63: 'copy', 65: 'cost', 67: 'count', 69: 'cut', 71: 'depend on', 73: 'destroy', 75: 'devote', 77: 'die', 79: 'different', 81: 'dislike', 83: 'do', 85: 'do badly', 87: 'do well', 89: 'draw', 91: 'drive', 93: 'enable', 95: 'endure', 97: 'equal', 99: 'expect', 101: 'fall', 103: 'feel', 105: 'fight', 107: 'fill', 109: 'find', 111: 'fit', 113: 'follow', 115: 'forget', 117: 'get', 119: 'get in', 121: 'give', 123: 'go', 125: 'grow', 127: 'handle', 129: 'harder than', 131: 'have', 133: 'hear', 135: 'help', 137: 'hit', 139: 'hold', 141: 'hurry', 143: 'hurt', 145: 'include', 147: 'involve', 149: 'join', 151: 'joke', 153: 'keep', 155: 'keep secret', 157: 'kill', 159: 'know', 161: 'lead', 163: 'learn', 165: 'leave', 167: 'light', 169: 'like', 171: 'limit', 173: 'live', 175: 'live in', 177: 'located', 179: 'lock out', 181: 'lose', 183: 'made of', 185: 'make', 187: 'mean', 189: 'meet', 191: 'miss', 193: 'mix', 195: 'more expensive than', 197: 'move', 199: 'move-object', 201: 'need', 203: 'None', 205: 'occupy', 207: 'open', 209: 'order', 211: 'organize', 213: 'own', 215: 'park', 217: 'pay attention', 219: 'play', 221: 'possible', 223: 'pray', 225: 'prepare', 227: 'prevent', 229: 'promise', 231: 'protect', 233: 'pull', 235: 'put', 237: 'rain', 239: 'raise', 241: 'reach', 243: 'read', 245: 'recycle', 247: 'relationship', 249: 'relax', 251: 'remember', 253: 'remove', 255: 'reserve', 257: 'return', 259: 'rule', 261: 'save', 263: 'see', 265: 'seek', 267: 'seem', 269: 'send', 271: 'sentiment-negative', 273: 'sentiment-positive', 275: 'share', 277: 'shoot', 279: 'show', 281: 'similar', 283: 'sing', 285: 'sit', 287: 'sleep', 289: 'smell', 291: 'smoke', 293: 'speak', 295: 'spend', 297: 'start', 299: 'stay', 301: 'stop', 303: 'survive', 305: 'take', 307: 'taste', 309: 'teach', 311: 'think', 313: 'tired', 315: 'try', 317: 'use', 319: 'visit', 321: 'wait', 323: 'wake up', 325: 'walk', 327: 'want', 329: 'waste', 331: 'wear', 333: 'win', 335: 'work', 337: 'worry', 339: 'write'}

    if MODEL == 'albert':
        # model = AlbertTripleExtractor('../../model_transformer/models/2022-04-27', bio_lookup)
        model = AlbertTripleExtractor('../../model_transformer/models/level1', bio_lookup_l1)
        # model = AlbertTripleExtractor('../../model_transformer/models/level2', bio_lookup_l2)
    else:
        raise Exception('model %s not recognized' % MODEL)

    result = evaluate(TEST_FILE, model, k=MIN_CONF, deduplication=False)

    # Save to file
    save_results(result, MODEL, TEST_FILE, MIN_CONF)
