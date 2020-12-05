# Neural Network + Genetic Algorithim to play the game Snake

Supervised Neural Networks require data, which would be messy to aquire in a virtual environment.
Using a Genetic Algorithim which is a Meta Heuristic method to optimize the weights for the Neural Network, we can teach the 
snake to find and eat food.

INPUTS:

    1. Binary (is there food to my left?)
    2. Binary (is there food ahead of me?)
    3. Binary (is there food to my right?)
    4. Distance to the wall
    5. Direction (either 1, 2, 3 or 4)
    
OUTPUT:

    1. Number representing "Should I turn left?"
    2. Number representing "Should keep going forward?"
    3. Number representing "Should I turn right?"
    
    Highest number gets picked.

TODO - Increase line of sight for snake for better turn predictions

Uses Eigen for Matrix Operations
https://github.com/PX4/eigen
