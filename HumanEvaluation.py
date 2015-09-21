"""
This class is used to store the human rankings
of the machine translated senteces that we use
as our gold standard.
It also provides functions to convert the results
of an evaluation score (like BLEU or ours)
to match the format of the human ranking.
"""
import unicodecsv
from operator import itemgetter
import operator

class HumanEvaluation:
    """ Store human ranking and evalute the results
    of evaluation systems against them."""

    scores_per_sentence = []
    # All scores in the range of this will be assigned
    # the same rank.
    equality_tolerance = 0.2

    def __init__(self, filename):
        with open(filename, 'rb') as csvfile:
            reader = unicodecsv.reader(csvfile, delimiter=',')
            firstline = True
            team_to_cnt = {}
            team_to_val = {}
            for row in reader:
                if not firstline:
                    score_order = []
                    team_order = []
                    for c in row:
                        if "de-en" in c:
                            team_order.append(c)
                    for score in row[-len(team_order):]:
                        score_order.append(int(score))
                    team_to_score = {}
                    for j in range(len(team_order)):
                        team_to_score[team_order[j]] = score_order[j]

                        if team_order[j] not in team_to_cnt:
                            team_to_cnt[team_order[j]] = 0
                        if team_order[j] not in team_to_val:
                            team_to_val[team_order[j]] = 0.0
                        team_to_cnt[team_order[j]] += 1
                        team_to_val[team_order[j]] += score_order[j]
                    self.scores_per_sentence.append(team_to_score)
                else:
                    firstline = False
            for team, val in team_to_val.items():
                print team, val / float(team_to_cnt[team])

    def get_rank_for_team(self, team, sentence):
        """ Give me the number of a sentence
        and a team name and I give you its human rank."""
        return self.scores_per_sentence[sentence][team]

    def get_teams(self, sentence):
        """ These teams have been ranked for the given sentence."""
        return list(self.scores_per_sentence[sentence].keys())

    def get_rank_for_sentence(self, sentence):
        """ Teams and their ranks for a given sentence. """
        return self.scores_per_sentence[sentence]

    def get_ranked_results(self, results):
        """ Convert scores s with 0 >= s <= 1 to
        ranks of human evaluation.
        Optimistic approach."""
        result_ranked = {}
        last_rank = len(results)+1
        last_score = 1000
        for team, score in sorted(results.iteritems(), key=itemgetter(1), reverse=True):
            if last_score - score <= self.equality_tolerance:
                result_ranked[team] = last_rank
            else:
                last_rank -= 1
                last_score = score
                result_ranked[team] = last_rank
        return result_ranked

    def get_ranked_resultsAlt(self, results):
        """ Fixed-range approach."""
        result_ranked = {}
        last_rank = len(results)+1
        last_score = 1000
        for team, score in sorted(results.iteritems(), key=itemgetter(1), reverse=True):
            if score > 0.0 and score <= 0.2:
                result_ranked[team] = 1
            if score > 0.2 and score <= 0.4:
                result_ranked[team] = 2
            if score > 0.4 and score <= 0.6:
                result_ranked[team] = 3
            if score > 0.6 and score <= 0.8:
                result_ranked[team] = 4
            if score > 0.8 and score <= 1:
                result_ranked[team] = 5
        return result_ranked

    def get_ranked_resultsPess(self, results):
        """ Pessimistic approach."""
        result_ranked = {}
        last_rank = 1
        last_score = 0
        for team, score in sorted(results.iteritems(), key=itemgetter(1), reverse=False):
            if score - last_score <= self.equality_tolerance:
                result_ranked[team] = last_rank
            else:
                last_rank += 1
                last_score = score
                result_ranked[team] = last_rank
        return result_ranked

    def compare_best(self, results, sentence):
        results_ranked = self.get_ranked_results(results)
        results_gold = self.get_rank_for_sentence(sentence)
        best_team, best_score = max(results_gold.iteritems(), key=operator.itemgetter(1))
        ref_score = results_ranked[best_team]
        return best_score - ref_score

    def compare(self, results, sentence):
        """ Compare the results of a sentence with the human ranking."""
        # rank results
        result_ranked = self.get_ranked_results(results)
        loss = 1.0 / len(self.get_teams(sentence))
        score = 1.0
        # print "calculated result:     ", result_ranked
        # print "human evaluated result:", self.get_rank_for_sentence(sentence)
        for team, rank in self.get_rank_for_sentence(sentence).iteritems():
            if team in result_ranked:
                if rank != result_ranked[team]:
                    score -= loss
            else:
                score -= loss
        return score

    def diffAll(self, sentence, res):
        """ Compare the results of a sentence with the
        human ranking and return a score for each team.
        This has been used in the final test."""
        # rank results
        ret = {}
        result_ranked = self.get_ranked_resultsPess(res)
        results_gold = self.get_rank_for_sentence(sentence)
        for team in result_ranked.iterkeys():
            ret[team] = abs(results_gold[team] - result_ranked[team])
        return ret
