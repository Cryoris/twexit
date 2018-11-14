from config.config import data_dir
import analyzer.analyzer as an
from loader.loader import Loader

ssix = an.SSIXAnalyzer(data_dir)
vader = an.VaderAnalyzer()

def test1():
    # Test case 1
    tweets = ["Terrorists want us to eat more fudge", 
              "But they don't want Jonas as he has no beard anymore",
              "Fuck that spineless moron cameron #mimimi",
              "Actually the EU is awesome #stay"]

    print ssix.get_values(tweets)
    print vader.get_values(tweets)

def test2():
    l = Loader(data_dir + "brexit_data.json")
    print "Total number of tweets: ", len(l.get_tweets())
    l.remove_retweets()
    print "Without retweets: ", len(l.get_tweets())

def test3():
    l = Loader("./data/weight.json")
    print l.get_dataframe()

test3()
