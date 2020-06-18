# Neural Network + Genetic Algorithim to play the game Snake

Supervised Neural Networks require data, and that would be non-trival to aquire in a virtual environment.
Using a Genetic Algorithim which is a Meta Heuristic method to optimize the weights for the Neural Network, We can teach the 
Snake to find and eat food.

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
