# Code for “Data”

> Data selecting & processing

### Code Running Order:

"1_filter.py"

"2_mix.py"

"3_rough_cmp.py"

"4_compare.py"

"5_split_dataset.py"

"6_filter_new_rule.py"

"7_generate_dataset_dur.py"

"7_generate_dataset_ON.py"



### Introduction:

**1_filter**: preliminarily filter original MIDIs

**2_mix**: mix the MIDI with two tracks into one track.

**3_rough_cmp**: find the simialr midis roughly

**4_compare**: based on the result of rough comparison, find the similar midis carefully with needleman

**5_split_dataset**: based on the result of comparison, spilt MIDIs to three parts (train, validation and test)

**6_filter_new_rule**: filter selected MIDIs in three datasets with new rules, which make the MIDI meet our requirements

**7_generate_datase_dur**: generate dataset for traning (with duration-map and key-map)

**7_generate_datase_ON**: generate dataset for traning (with ON-map and key-map)

**process_func**: a Python file includes necessary function for data selecting and processing.

**find_seq.c/find_seq.so**: a function for MIDI processing.



### Notice: 

1. We do this research step by step, so we do not consider all necessary rules for selecting MIDIs at first, which leads to two separate steps for selecting MIDIs. This needs to be improved in the future. 
2. There are three data representations for our model training and each model has its own data representation. They are key-map dataset , duration-map and key-map dataset, ON-map and key-map dataset. The key-map dataset can be extracted from duration-map and key-map dataset.
3. Several hyperparameters need to be modified in the first 7 Python file like the original MIDI document path.






# Code for "trian"

> Model construction & training

### key

key.py only uses key-map data for training.

### duration_and_key

duration_and_key.py uses duration-map and key-map for training.

### ON_and_key

ON_and_key.py uses ON-map and key-map for training.