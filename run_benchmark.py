import os
#from subprocess import call
path = os.path.dirname(__file__)
task_sets = ['hvm_basic_categorization']
base_command = 'sbatch -n 10 -c 2 benchmark_test_set.sh '
for task_set in task_sets:
    command = base_command+task_set
    os.system(command)
