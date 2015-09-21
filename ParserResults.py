"""
This class reads in the results of the SEMAFOR parser as JSON strings
and provides them for easy access from the other classes.
"""
import json
import unicodecsv

class ParserResults:
    """
    This class reads in the results of the SEMAFOR parser as JSON strings
    and provides them for easy access from the other classes.
    """
    table = []
    gold = 'newstest2014-deen-ref.de-en'

    def __init__(self, datafile):
        with open(datafile, 'rb') as csvfile:
            reader = unicodecsv.reader(csvfile, delimiter='\t')
            firstline = []
            for row in reader:
                if len(firstline) == 0:
                    firstline = row
                else:
                    row_data = {} # team -> json
                    for i in range(len(row)):
                        team = firstline[i]
                        parse_result = json.loads(row[i].encode('utf-8'))
                        row_data[team] = parse_result
                    self.table.append(row_data)

    def get_gold(self):
        """ The name of the column that serves as the reference team."""
        return self.gold

    def get_row(self, row):
        """ Get the sentences in this row."""
        return self.table[row]

    def get_row_for_teams(self, list_of_teams, row):
        """ Give me only the sentences for the teams I give you.
        This is useful since not all sentences have been ranked
        by humans."""
        ret = {}
        for team in list_of_teams:
            if team in self.table[row]:
                ret[team] = self.table[row][team]
        return ret

    def get_sentence(self, team, row):
        """ Get a single JSON object for a row and a team."""
        return self.get_sentence_for_object(self.table[row][team])

    def get_sentence_for_object(self, jsonobject):
        """ Return the acutal sentence as a string."""
        return " ".join(jsonobject['tokens']).encode('utf-8')

    def get_number_of_rows(self):
        """ Number of sentences."""
        return len(self.table)
