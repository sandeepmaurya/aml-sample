import utils

run = utils.run_experiment()
run_details = run.get_details()
print(f'Finished run: {run_details["runId"]}')
print(f'Run Details:\n{run.get_details()}')
print(f'\nMetrics:\n{run.get_metrics()}')
