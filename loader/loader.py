import json
import csv
import pandas
import os
import StringIO

class Loader:

    data = []

    def __init__(self, filename):
        self.load_file(filename)

    def load_file(self, filename):
        try: 
            with open(filename, "rb") as handler:
                content = handler.read()
        except:
            raise SystemError("Couldn't open file " + filename)

        io = StringIO.StringIO(content)
        suffix = os.path.splitext(filename)[1]
        if suffix == ".json":
            self.data = pandas.read_json(io)
        elif suffix == ".csv":
            self.data = pandas.read_csv(io, error_bad_lines=False)
        else:
            raise NotImplementedError("Only .json or .csv files can be loaded!")

        self.remove_deleted()
        self.to_lower()

        return self

    def to_lower(self):
        self.data["text"] = self.data["text"].apply(lambda x: x.lower())

    def remove_retweets(self):
        rt = lambda x: x[:2] == "rt"
        self.data = self.data[self.data["text"].apply(rt) == False]

    def remove_deleted(self, colname="text"):
        self.data = self.data[self.data["text"] != "deleted"]

    def get_tweets(self, colname="text"):
        return list(self.data[colname])

    def get_dataframe(self):
        return self.data