"""

Find abbreviations and their long forms that appear in LRABR or ADAM but not both.
Use this for train/test sets for acronym LSTM-CTC

"""

import re

keep_abbr_re = "[\\w.\\-/&\\s]+"

adam = [line.strip().split('\t') for line in open('adam_database') if not line.startswith('#')]
lrabr = [line.strip().split('|')[:-1] for line in open('LRABR')]

adam_dict = {}
lrabr_dict = {}

adamlongs = []
for x in adam:
    senses = x[2].split('|')
    senses = [sense[:sense.find(':')] for sense in senses]
    adamlongs += senses
    for sense in senses:
        if sense not in adam_dict and re.match(keep_abbr_re, x[0]):
            adam_dict[sense] = x[0]

lrabrlongs = [x[4] for x in lrabr]
used_euis = {}
for x in lrabr:
    if x[4] not in lrabr_dict and re.match(keep_abbr_re, x[1]):
        lrabr_dict[x[4]] = x[1]


unique_long_adam = set(adamlongs)
unique_long_lrabr = set(lrabrlongs)

print len(unique_long_adam)
print len(unique_long_lrabr)

print len(adam_dict)
print len(lrabr_dict)

adam_only = {x:y for x,y in adam_dict.iteritems() if x not in lrabr_dict}
lrabr_only = {x:y for x,y in lrabr_dict.iteritems() if x not in adam_dict}

print len(adam_only)
print len(lrabr_only)

with open('adam_only.txt', 'w') as w:
    for a, l in adam_only.iteritems():
        w.write(l)
        w.write('\n')
        w.write(a)
        w.write('\n')

with open('lrabr_only.txt', 'w') as w:
    for a, l in lrabr_only.iteritems():
        w.write(l)
        w.write('\n')
        w.write(a)
        w.write('\n')
