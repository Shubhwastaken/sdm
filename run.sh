#!/bin/bash
# Shell script to run the RL Scheduling Simulator on Linux/Mac

echo "========================================"
echo " RL Scheduling Simulator"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "Python found!"
echo ""

# Check if requirements are installed
echo "Checking dependencies..."
python3 -c "import numpy, pandas, matplotlib, seaborn, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "Some required packages are missing."
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo ""
        echo "Error: Failed to install dependencies"
        exit 1
    fi
fi

echo "Dependencies OK!"
echo ""

# Ask user what to run
echo "What would you like to run?"
echo "  1. Quick Demo"
echo "  2. Full Simulation"
echo "  3. Exit"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Running Quick Demo..."
        echo ""
        python3 demo.py
        ;;
    2)
        echo ""
        echo "Running Full Simulation..."
        echo ""
        cd src
        python3 main.py
        cd ..
        ;;
    3)
        echo ""
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo ""
        echo "Invalid choice. Running Quick Demo by default..."
        echo ""
        python3 demo.py
        ;;
esac

echo ""
echo "========================================"
echo " Execution Complete"
echo "========================================"
