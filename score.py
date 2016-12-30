import ast
import re

stopwords = {
    'and',
    'of',
    'the',
    'an',
    'for',
    'in'
}

def simple_rule_based_acr(longform):
    words = re.split("[ \\-&/]", longform)
    abbr = ''.join([w[0] for w in words if len(w) and w.lower() not in stopwords])
    return abbr.upper()

def strip_underscores(abbr):
    return abbr.replace('_','')

lastline = ''
for line in open('dev_log.txt'):
    lastline = line

tups = ast.literal_eval(lastline)

print len(tups)

rule_correct = [x[1] == simple_rule_based_acr(x[0]) for x in tups]
lstm_correct = [x[1] == strip_underscores(x[2]) for x in tups]


same_guess = [simple_rule_based_acr(x[0]) == strip_underscores(x[2]) for x in tups]


print [val for ind, val in enumerate(tups) if lstm_correct[ind] and not rule_correct[ind]]

print '~~~~~~~~~~~~~ LSTM ONLY CORRECT'
for ind in range(len(tups)):
    if lstm_correct[ind] and not rule_correct[ind]:
        print tups[ind][0], '\t', tups[ind][1], '\t', simple_rule_based_acr(tups[ind][0]), '\t', strip_underscores(tups[ind][2])


print '~~~~~~~~~~~~~ RULE ONLY CORRECT'

for ind in range(len(tups)):
    if rule_correct[ind] and not lstm_correct[ind]:
        print tups[ind][0], '\t', tups[ind][1], '\t', simple_rule_based_acr(tups[ind][0]), '\t', strip_underscores(tups[ind][2])

print sum(rule_correct)
print sum(lstm_correct)

print 'both correct', sum([x and y for x,y in zip(rule_correct, lstm_correct)])
print 'neither correct', sum([not x and not y for x,y in zip(rule_correct, lstm_correct)])
print 'rule only correct', sum([x and not y for x,y in zip(rule_correct, lstm_correct)])
print 'lstm only correct', sum([not x and y for x,y in zip(rule_correct, lstm_correct)])
print 'both systems same guess', sum(same_guess)

