

generate_lasso_data (generates lasso samples, with and without noise) - how the model-free classes expect them.

scale_test_dm_rnn_enhanced.sh (change name to something more fitting) - trains the model-free-l2o models l2o-DM, l2o-rnnprop and their enhanced versions on different (M, N) lasso-problems.

run_scaling_test.sh - does the same as scale_test_dm_rnn_enhanced, but for the model-free l2o models.

plot_figure_single - plots the lasso expermiment like done in figure 6 of the paper but just for a single model.

plot_figure6.py - plots the lasso expermiment like done in figure 6 of the paper for multiple models.

train_and_plot_figure6.sh - trains models on the lasso problem and plots the results on a figure like figure 6 in the paper.

gpu_mem_sampler - a class that during some set time interval, checks gpu usage and memory cost of the process (adapted for consumer GPU) Stores the mean and median GPU usage as well as total time consumed.

