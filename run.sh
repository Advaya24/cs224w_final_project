
#!/bin/bash

# Define the values of num_layers to iterate over
num_layers_values=(1 2 4 8)

# Iterate over the num_layers values
for num_layers in "${num_layers_values[@]}"
do
    echo "Running train_lightgcn for num_layers = $num_layers"
    python train_lightgcn.py --num_layers $num_layers
done
