*********************************
Run Order:
"filter.py"
"mix.py"
"rough_cmp.py"
"compare.py"
"generate_dataset.py"

*********************************
Introduction:
filter: filter original MIDIs that meet our requirements.

mix: mix the MIDI with two tracks into one track.

rough_cmp: find the simialr midis roughly

compare: based on the result of test_c, find the similar midis carefully with needleman and search

generate_dataset: divid midis into three parts and generate dataset for traning
*********************************
Notice: 
"filter.py" will generate path file for all MIDIs.

"mix.py" will generate path files for filtered MIDIs and mixed MIDIs

"rough_cmp.py" will use 60 size window which is located at 100 to 160 of a sequence
to search another a sequence.

"compare.py" searchs one whole midi sequence with another short whole sequence 
and uses needleman to find similar sequence, which is based on the result of 
"rough_cmp.py".

  
*********************************
