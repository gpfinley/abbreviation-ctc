# abbreviation-ctc

An attempt to model how humans construct abbreviations using bidirectional long short-term memory networks and connectionist temporal classification. Currently set up to learn biomedical abbreviations (train on UMLS and test on non-UMLS abbreviations from ADAM).

Uses Python and TensorFlow. Code is still very much in the experimental stage! It's ugly, but it kind of works. A trained network (couple of days, on GPUs) achieves better performance than a rule-based initialism baseline. Lots more to do with tweaking the network and looking at different data sets.
