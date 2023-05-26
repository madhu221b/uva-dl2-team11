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


