## Accuracy and F1 experiments

To run the experiments that relate accuracy and f1 score to  experiments:
* navigate to the directory containing noising_experiment.py.
* Set up the config files, as was required for model training.
* Run the following terminal command to generate the raw data

``` python noising_experiment.py  --cfg <path_to_config_file> --output_file <path_to_file_to_save_results_to> --device <cuda_device> --num_graphs <number_of_graphs_to_generate_results_for>```

* Upon completion, the program will dump a pickled dataframe containing the results.
* Load that dataframe in the notebook 'noiser_results.ipynb' using
```
df = pre_process_df(output_file_path)
ra = get_relative_accuracy(df)
f1 = get_relative_f1s(df)  
```

There are two pre-prepared pickle files available for use. They are found in ```./assets/noising_experiments/<model_name>.pkl ```

## Influence Score experiments

To obtain run the influence score experiments:
* navigate to the directory containing model_inference.py.
* Set up the config files, as was required for model training.
* Run the following terminal command to generate the raw gradient information:

```
python model_inference.py --cfg <path_to_config_file> device 'cuda_device'
```

* This will output a pickle file containing the gradients to "inf_scores_{model_type}.pkl".
* Now run the following code to plot the influence score for a given model:

```
from src.influence import process_all_graphs, plot_mean_influence_by_distance
import matplotlib.pyplot as plt

file_name = './path/to/pickle/
influence_df_gcn = process_all_graphs('inf_scores_gcn_with_adj.pkl', normalise=True)

fig, ax = plt.subplots()
plot_mean_influence_by_distance(influence_df_gcn, ax, 'GCN')

ax.set_xlabel('Shortest path distance from target node')
ax.set_ylabel('Proportion of total gradient')
ax.legend()
```

Alternatively, the notebook 'influence.ipynb' contains an example of this pipeline.

For convenience, pickle files containing the influence scores of 3 models are avaiable in

```assets/influence_experiments/<model_name>.pkl```

