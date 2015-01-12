import os
#from subprocess import call
path = os.path.dirname(__file__)
task_sets = ['hvm_basic_categorization']
script_location = os.path.join(path, 'benchmark_task_set.sh')
base_command = 'sbatch -n 10 -c 2 %s ' % script_location
for task_set in task_sets:
    command = base_command+task_set
    os.system(command)
