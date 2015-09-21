"""
A proof-of-concept that semantic based evaluation
of the quality of machine translated sentences works.

Authors: Katarina Krueger, Johannes Gontrum
"""

import unicodecsv as csv
import nltk.align.bleu_score as bleu_score
import numpy as np
from ParserResults import ParserResults
from HumanEvaluation import HumanEvaluation
from nltk.corpus import wordnet as wn
from scipy.optimize import differential_evolution

class Penalties:
    """ Define all penalties or scores at one place for better parameter tuning """
    # <Text and Words>
    # When comparing words, a lexical or synonym match gets these scores
    score_for_matched_lexical = 1.0 # >= 0, <= 1
    score_for_matched_synonym = 1.0 # >= 0, <= 1

    word_window_left = 10 # > 0
    word_window_right = 10 # > 0
    factor_word_offset_penalty = 0.2 # >= 0, <= 1

    factor_sentence_length_mismatch = 0.6 # >= 0, <= 1
    # </Text and Words>
    # <Frame and Frame Elements>
    factor_name_mismatch = 0.1 # >= 0, <= 1

    fe_window_left = 10
    fe_window_right = 10
    factor_fe_offset_penalty = 1.0

    weight_target_frame_element = 3
    weight_frame_elements = 1
    # </Frame and Frame Elements>
    # <Sentence Level>
    frame_window_left = 10
    frame_window_right = 10
    factor_frame_offset_penalty = 1.0
    # </Sentence Level>


class SemMTEval:
    """ This class performes the semantic MT evaluation."""
    pen = Penalties()       # A bundle of parameters to fine-tune
    SYNSET_CACHE = {}       # A cache for the synonyms to speed things up
    INTERSECTION_CACHE = {} # Another cache to save intersection operations

    def __init__(self, synonyme_cache, intersection_cache, data, evaluator):
        """ Initialize the evaluator with given caches, an object that
        contains the data and an object to evaluate the calculated scores
        against the human ones. By default, you should use empty caches,
        but it is also possible to load or save caches from file."""
        self.SYNSET_CACHE = synonyme_cache
        self.INTERSECTION_CACHE = intersection_cache
        self.pen = Penalties()
        self.evaluator = evaluator
        self.data = data

    def get_synset(self, word):
        """ Enables caching for synset access"""
        #word = word.decode('utf-8')
        if word in self.SYNSET_CACHE:
            return self.SYNSET_CACHE[word]
        else:
            synset = set(wn.synsets(word))
            self.SYNSET_CACHE[word] = synset
            return synset

    def get_intersection(self, tup):
        """ Caches intersection of synsets """
        if tup in self.INTERSECTION_CACHE:
            return self.INTERSECTION_CACHE[tup]
        else:
            synset1 = self.get_synset(tup[0])
            synset2 = self.get_synset(tup[1])
            common = len(synset1.intersection(synset2)) > 0
            self.INTERSECTION_CACHE[tup] = common
            self.INTERSECTION_CACHE[(tup[1],tup[0])] = common
            return common

    def are_words_synonym(self, word1, word2):
        """ Returns True, if both words share a synonym set. """
        return self.get_intersection((word1, word2))

    def get_word_score_in_window(self, gold, candidate, use_synonyms, index):
        """Find the best matching word in the candidate sentence.
        These words can occure in a window of a few index left and right,
        but any offset will come with a penalty."""
        best_score = 0
        for offset in range(self.pen.word_window_left * -1, self.pen.word_window_right + 1):
            new_index = index + offset
            # iterate over the window
            if new_index >= 0 and new_index < len(gold) and new_index < len(candidate):
                if gold[new_index] == candidate[new_index]:
                    # it is a lexical match
                    new_score = self.pen.score_for_matched_lexical
                    new_score -= (abs(offset) * self.pen.factor_word_offset_penalty)
                    best_score = max(best_score, new_score)
                else: # maybe we find a semantic match
                    if use_synonyms and self.are_words_synonym(gold[index], candidate[index]):
                        # the words are synonymes
                        new_score = self.pen.score_for_matched_synonym
                        new_score -= (abs(offset) * self.pen.factor_word_offset_penalty)
                        best_score = max(best_score, new_score)
        return best_score

    def get_text_matches_window(self, gold, candidate, use_synonyms):
        """ Compares two lists of strings item per item and returns the number of matches.
        If the candiadate list is smaller than the gold list,
        only the words in gold are concidered. """
        matches = 0
        score = 0
        length = min(len(gold), len(candidate))
        for index in range(length):
            new_score = self.get_word_score_in_window(gold, candidate, use_synonyms, index)
            score += new_score
            if new_score > 0:
                matches += 1
        return {"matches" : float(matches), "score" : float(score)}


    def get_text_score(self, gold, candidate):
        """ Returns the score of a text as string. """
        length_of_gold = float(len(gold.split()))
        gold_tok = gold.split()
        candidate_tok = candidate.split()
        length_penalty = abs(length_of_gold-len(candidate_tok))
        length_penalty *= self.pen.factor_sentence_length_mismatch
        # Best case: both texts are lexically equal and in the same order
        if gold_tok == candidate_tok:
            return 1.0
        # Check with offset synonyms
        new_result = self.get_text_matches_window(gold_tok, candidate_tok, True)
        new_score = new_result['score']
        if (length_of_gold) > 0:
            new_score /= (length_of_gold)
        new_score -= length_penalty
        # make sure its not below zero
        new_score = max(0, new_score)
        return new_score

    def get_frame_element_score(self, fe1, fe2):
        """ Returns the score of two frame elements by comparing the text
        and weighting it depending on a name match."""
        name_match = fe1['name'] == fe2['name']
        score = self.get_text_score(fe1['spans'][0]['text'], fe2['spans'][0]['text'])
        if not name_match:
            score *= self.pen.factor_name_mismatch
        return score

    def get_frame_element_score_in_window(self, gold, candidate, index):
        """ Find the matching frame element and alow it to be within
        a certain offset."""
        best_score = 0
        for offset in range(self.pen.fe_window_left * -1, self.pen.fe_window_right + 1):
            new_index = index + offset
            if new_index > 0 and new_index < len(gold) and new_index < len(candidate):
                new_score = self.get_frame_element_score(gold[new_index], candidate[new_index])
                if gold[new_index] != candidate[new_index]:
                    new_score -= (abs(offset) * self.pen.factor_fe_offset_penalty)
                best_score = max(best_score, new_score)
        return best_score

    def get_frame_element_matches_window(self, gold, candidate):
        """ Call the find-in-window method for each frame element. """
        score = 0
        length = min(len(gold), len(candidate))
        if length == 0:
            return 0
        for index in range(length):
            score += self.get_frame_element_score_in_window(gold, candidate, index)
        return score

    def get_frame_score(self, goldframe, candidateframe):
        """ Compares two frames. """
        # Compare the target of both frames
        target_score = self.get_frame_element_score(goldframe['target'], candidateframe['target'])
        # Now let's check the frame elements!
        gold_fe = goldframe['annotationSets'][0]['frameElements']
        candidate_fe = candidateframe['annotationSets'][0]['frameElements']
        score_fe = self.get_frame_element_matches_window(gold_fe, candidate_fe)
        # Calculate the weighted average between the target match and
        # all other frame elements.
        score = (target_score * self.pen.weight_target_frame_element)
        score += (score_fe * self.pen.weight_frame_elements)
        score /= (self.pen.weight_frame_elements + self.pen.weight_target_frame_element)
        return score


    def get_frame_score_in_window(self, gold, candidate, index):
        """ Find the matching frame in a certain window. """
        best_score = 0
        for offset in range(self.pen.frame_window_left * -1, self.pen.frame_window_right + 1):
            new_index = index + offset
            if new_index > 0 and new_index < len(gold) and new_index < len(candidate):
                new_score = self.get_frame_score(gold[new_index], candidate[new_index])
                if gold[new_index] != candidate[new_index]:
                    new_score -= (abs(offset) * self.pen.factor_frame_offset_penalty)
                best_score = max(best_score, new_score)
        return best_score

    def get_sentence_score(self, gold_sentence, candidate_sentence):
        """ Calculate the score for the whole sentence by comparing
        all frames within them. This will go on recursivly to the
        frame elements and to the words themselves."""
        # Collect all frames
        gold_frames = gold_sentence['frames']
        candidate_frames = candidate_sentence['frames']
        # Check first in the actual order
        score = 0
        length = min(len(gold_frames), len(candidate_frames))
        if length == 0:
            return 0.0
        for index in range(length):
            score += self.get_frame_score_in_window(gold_frames, candidate_frames, index)
        score /= float(length)
        return min(score, 1.0)

    def run_compare(self):
        """ Compare all the sentences in the given data and return
        the median of the difference between the human ranking and
        the calculated one."""
        misses = []
        for row in range(self.data.get_number_of_rows()):
            ref_sentence = self.data.get_row(row)[self.data.get_gold()]
            results = {}
            for team, team_sentence in self.data.get_row_for_teams(self.evaluator.get_teams(row), row).iteritems():
                results[team] = self.get_sentence_score(ref_sentence, team_sentence)
            misses.append(self.evaluator.compare_all(results, row))
        print np.median(misses), np.mean(misses)
        return np.median(misses)

    def run(self, args):
        """Give this method all values for the penalties and scores,
        and it will calculate the average mean miss. This is a
        great method to use a multi variate optimization function
        to find the best values for the penalities class."""
        self.pen.score_for_matched_lexical = args[0]
        self.pen.score_for_matched_synonym = args[1]
        self.factor_word_offset_penalty = args[2]
        self.factor_sentence_length_mismatch = args[3]
        self.factor_name_mismatch = args[4]
        self.factor_fe_offset_penalty = args[5]
        self.weight_target_frame_element = args[6]
        self.weight_frame_elements = args[7]
        self.factor_frame_offset_penalty = args[8]
        misses = []
        for row in range(self.data.get_number_of_rows()):
            ref_sentence = self.data.get_row(row)[self.data.get_gold()]
            results = {}
            for team, team_sentence in self.data.get_row_for_teams(self.evaluator.get_teams(row), row).iteritems():
                results[team] = self.get_sentence_score(ref_sentence, team_sentence)
            misses.append(self.evaluator.compare_all(results, row))
        return np.mean(misses) / 5.0

    def runAndSave(self, args):
        """ This method was build upon the normal 'run' method
        but was used for generating the final results. The team
        names are hardcoded to match columns in the CSV files,
        so this is rather an example how to perform a detailed
        analysis."""
        self.pen.score_for_matched_lexical = args[0]
        self.pen.score_for_matched_synonym = args[1]
        self.factor_word_offset_penalty = args[2]
        self.factor_sentence_length_mismatch = args[3]
        self.factor_name_mismatch = args[4]
        self.factor_fe_offset_penalty = args[5]
        self.weight_target_frame_element = args[6]
        self.weight_frame_elements = args[7]
        self.factor_frame_offset_penalty = args[8]

        team_to_row = { "newstest2014.CMU.3461.de-en" : 0,
                        "newstest2014.DCU-ICTCAS-Tsinghua-L.3444.de-en" : 1,
                        "newstest2014.LIMSI-KIT-Submission.3359.de-en" : 2,
                        "newstest2014.RWTH-primary.3266.de-en" : 3,
                        "newstest2014.eubridge.3569.de-en" : 4,
                        "newstest2014.kit.3109.de-en" : 5,
                        "newstest2014.onlineA.0.de-en" : 6,
                        "newstest2014.onlineB.0.de-en" : 7,
                        "newstest2014.onlineC.0.de-en" : 8,
                        "newstest2014.rbmt1.0.de-en" : 9,
                        "newstest2014.rbmt4.0.de-en" : 10,
                        "newstest2014.uedin-syntax.3035.de-en" : 11,
                        "newstest2014.uedin-wmt14.3025.de-en" : 12,
                        "newstest2014-deen-ref.de-en" : 13}
        teams = list(team_to_row.keys())
        teams.remove("newstest2014-deen-ref.de-en")
        def_list = ['-' for x in range(len(team_to_row))]
        with open('ourPessimisticRankingDiff.csv', 'wb') as our_csvfile:
            with open('bleuPessimisticRankingDiff.csv', 'wb') as bleu_csvfile:
                ourwriter = csv.writer(our_csvfile)
                bleuwriter = csv.writer(bleu_csvfile)

                our_print_res = list(def_list)
                bleu_print_res = list(def_list)
                for team in team_to_row.iterkeys():
                    if team in teams:
                        our_print_res[team_to_row[team]] = team
                        bleu_print_res[team_to_row[team]] = team

                ourwriter.writerow(our_print_res)
                bleuwriter.writerow(bleu_print_res)

                for row in range(self.data.get_number_of_rows()):
                    print row
                    ref_sentence = self.data.get_row(row)[self.data.get_gold()]
                    our_print_res = list(def_list)
                    bleu_print_res = list(def_list)
                    our_results = {}
                    bleu_results = {}
                    for team, team_sentence in self.data.get_row_for_teams(self.evaluator.get_teams(row), row).iteritems():
                        our = self.get_sentence_score(ref_sentence, team_sentence)
                        our_results[team] = our
                        bleus = bleu_score.bleu(self.data.get_sentence_for_object(team_sentence).split(), self.data.get_sentence_for_object(ref_sentence).split(), [1])
                        bleu_results[team] = bleus

                    for team, rank in self.evaluator.diffAll(row, our_results).iteritems():
                        our_print_res[team_to_row[team]] = rank

                    for team, rank in self.evaluator.diffAll(row, bleu_results).iteritems():
                        bleu_print_res[team_to_row[team]] = rank

                    ourwriter.writerow(our_print_res)
                    bleuwriter.writerow(bleu_print_res)


if __name__ == "__main__":
    # This is how our class can be used:
    e = SemMTEval({}, {}, ParserResults('trainingParsed.tsv'), HumanEvaluation('trainingRating.csv'))
    print e.run_compare()
    # With these values the final results were generated:
    e.runAndSave([0.04011861, 0.5604594, 0.37964945, 0.49765078, 0.55329068, 0.39652122, 23.52352186, 21.10976504, 0.33117128])
    # An example how Scikit can be used to find optimal parameters:
    print differential_evolution(e.run, [(0.00, 1.0), (0.00, 1.0), (0.00, 1.0), (0.00, 1.0), (0.00, 1.0), (0.00, 1.0), (1, 1), (1, 100), (0.00, 1.0)], strategy='rand2bin')
