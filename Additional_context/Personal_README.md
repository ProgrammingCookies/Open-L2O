## ../benchmarking/core/method.py

The abstract class for methods to be benchmarked. Each method should inherit from this class and implement the required methods.

## ../benchmarking/core/runner.py

The abstract class for running benchmark experiments. Each runner should inherit from this class and implement the required methods.


## ../benchmarking/core/result.py

The abstract class for results of benchmarked methods. Each result should inherit from this class and implement the required methods.
leaderboard is summary of results from different methods.


# ../benchmarking/core/core.py

Has help functions to make sure the scripts for the different methods are run in the correct environment. 
It also has helpers to make sure the results are saved in the correct format.
It also makes it so when multiple ones are run in the same script (which is another way to do it) multiple imports don't collide with each other in sys.modules.





