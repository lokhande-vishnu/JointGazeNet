#################################################################################
# for train -- source, source labels, target, target_no_labels
# concats 'source' and 'target'-- passes through network -- gets back source_fc7, target_fc7
# FC: source_fc7 to source_fc8, target_fc7 to target_fc8
# loss 1:regssn: source_fc8 and source_labels
# loss 2:mmd: source_fc8 and target_fc8

# for testing --  target, target_labels
# calls 'target' as data -- passes through network -- gets back source_fc7
# FC: source_fc7 to source_fc8
# accuracy: fc8 and labels
#################################################################################
# for train -- source_data 200, source_lables, target_dataL 50, target_labelsL, target_dataN 150
# concats [a; b; c] --- passes through network -- gets back [a_fc7; b_fc7; c_fc7]
# FC:  a_fc7 to a_fc8, [b_fc7; c_fc7] to [b_fc8; c_fc8]
# loss 1: regre: a_fc8 and sourc_labesl
# loss 2: regre: b_fc8 and target_labels
# loss 3: mmd: a_fc7 and [b_fc7; c_fc7]

# For testing it is the same

#################################################################################
# for train -- source_data 200, source_lables, target_dataL 50, target_labelsL, target_dataN 150
# concats [a; b; c] --- passes through network -- gets back [a_fc7; b_fc7; c_fc7]
# a -- passes through private network -- gets back [pv_a_fc7]
# [b;c] -- passes through private network -- getback [pv_b_fc7;pv_c_fc7]


# FC:  a_fc7 to a_fc8, [b_fc7; c_fc7] to [b_fc8; c_fc8]
# loss 1: regre: a_fc8 and sourc_labesl
# loss 2: mmd: b_fc8 and target_labels
# loss 3: mmd: a_fc7 and [b_fc7; c_fc7]
# loss 4: diff: a_fc7 and pv_a_fc7
# loss 5: diff: [b_fc7; c_fc7] and [pv_b_fc7;pv_c_fc7] 

#################################################################################
