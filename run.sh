#!/bin/bash

# Function to check the exit code and proceed or exit
check_exit_code() {
    if [ $1 -eq 0 ]; then
        echo "Success!"
    else
        echo "Error: Script failed with exit code $1"
        exit $1
    fi
}

# Step 1: Run data_loader.py
echo "Running data_loader.py..."
python data_loader.py
check_exit_code $?

# Step 2: Run model_builder.py
echo "Running model_builder.py..."
python model_builder.py
check_exit_code $?

# Step 3: Run train.py
echo "Running train.py..."
python train.py
check_exit_code $?

# Step 4: Run evaluation.py
echo "Running evaluation.py..."
python evaluation.py
check_exit_code $?

echo "All scripts executed successfully!"
