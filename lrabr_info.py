from __future__ import division
import re
import editdistance

__author__ = 'gpfinley'

abbrs = []

for line in open('/Users/gpfinley/LRABR'):
    try:
        eui, abbr, type, abbreui, longform, _ = line.split('|')
    except:
        continue
    abbrs.append((type, abbr.replace('.',''), longform))

print len(abbrs), 'abbreviations total'
print len([x for x in abbrs if x[0] == 'acronym']), 'of these are acronyms'
print max([len(x[1]) for x in abbrs])

import sys
sys.exit(0)


split_on_re = "[\\s\\-/&]"

perfect_acronyms = []
edit_distances = []
for abbrtup in abbrs:
    if abbrtup[0] == 'acronym':
        perf_acr = ''.join([w[0] for w in re.split(split_on_re, abbrtup[2]) if len(w)])
        edit_distances.append(editdistance.eval(perf_acr, abbrtup[1]))
        if perf_acr.lower() == abbrtup[1].lower():
            perfect_acronyms.append(abbrtup)
        else:
            print perf_acr, abbrtup[1], abbrtup[2], re.split(split_on_re, abbrtup[2])

print len(perfect_acronyms)
print sum(edit_distances) / len(edit_distances)


