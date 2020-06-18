#include <SFML/Graphics.hpp>
#include "GNN.h"

sf::RenderWindow window(sf::VideoMode(800, 600), "Snake Algo");
sf::RectangleShape bodypart(sf::Vector2f(20, 20));

sf::Vector2f food(505, 400);
sf::RectangleShape foodRect(sf::Vector2f(20, 20));

int dir = 0;

//SPEED SHOULD BE ABLE TO DIVIDE WIDTH AND LENGTH
float speed = 5;

struct Snake
{
	std::vector<sf::Vector2f> parts = { sf::Vector2f(400, 300), sf::Vector2f(405, 300), sf::Vector2f(410, 300), sf::Vector2f(415, 300), sf::Vector2f(420, 300) };
	int score = 0;
	bool dead = false;

	int time = 0;
	int energy = 500;
};

bool eatFood(sf::Vector2f snake)
{
	bodypart.setPosition(snake);

	if (bodypart.getGlobalBounds().intersects(foodRect.getGlobalBounds()) )
	{
		return true;
	}

	return false;
}

void drawSnake(Snake &snake)
{
	for (sf::Vector2f &i : snake.parts)
	{
		bodypart.setPosition(i);
		window.draw(bodypart);
	}
}

void gameLogic(Snake& snake, GNN::Solution &sol)
{
	for (int i = snake.parts.size() - 1; i > 0; i--)
	{
		snake.parts[i] = snake.parts[i - 1];
	}

	if (dir == 2)
	{
		snake.parts[0].x += speed; //right
	}
	else if (dir == 0)
	{
		snake.parts[0].x -= speed; //left
	}
	else if (dir == 1)
	{
		snake.parts[0].y += speed; // down
	}
	else if (dir == 3)
	{
		snake.parts[0].y -= speed; // up
	}

	if (eatFood(snake.parts[0]))
	{
		snake.parts.push_back(sf::Vector2f(-100, -100));

		snake.score++;

		food.x = std::rand() % 780 + 10;
		food.y = std::rand() % 580 + 10;

		snake.energy += 400;

		foodRect.setPosition(food);
	}
}

bool colliding(Snake snake, GNN::Solution &sol)
{
	for (int i = 0; i < snake.parts.size(); i++)
	{
		for (int j = 0; j < snake.parts.size(); j++)
		{
			//checking if same object
			if (&snake.parts[i] != &snake.parts[j])
			{
				//checking is same location
				if (snake.parts[i] == snake.parts[j])
				{
					return true;
				}
			}
		}
	}

	if (780 < snake.parts[0].x || snake.parts[0].x < 0)
	{
		return true;
	}
	if (580 < snake.parts[0].y || snake.parts[0].y < 0)
	{
		return true;
	} 

	return false;
}

void playSolution(GNN::Solution sol)
{
	Snake snek;

	sf::Event eve;
	window.setFramerateLimit(120);

	while (!snek.dead)
	{

		window.pollEvent(eve);
		window.clear();

		//SPLIT DRAW SNAKE METHOD
		drawSnake(snek);
		window.draw(foodRect);

		window.display();


		window.pollEvent(eve);


		//base game, collison and logic
		gameLogic(snek, sol);

		snek.time += 1;

		if (colliding(snek, sol))
		{
			snek.dead = true;
		}

		if (snek.energy <= 0)
		{
			snek.dead = true;
		}

		snek.energy -= 1;

			Eigen::Matrix<float, 1, 4> inputvals;
		inputvals << food.x - snek.parts[0].x, food.y - snek.parts[0].y, dir + 1, 0;

		if (dir == 0)
		{
			inputvals(3) = snek.parts[0].x;
		}
		else if (dir == 2)
		{
			inputvals(3) = 780 - snek.parts[0].x;
		}

		if (dir == 3)
		{
			inputvals(3) = snek.parts[0].y;
		}
		else if (dir == 1)
		{
			inputvals(3) = 600 - snek.parts[0].y;
		}

		//Movement Prediction
		Eigen::Vector3f movePred = GNN::feed(sol, inputvals);

		if (movePred(0) > movePred(1))
		{
			if (movePred(0) > movePred(2))
			{
				//left
				if (dir == 3)
				{
					dir = 0;
				}
				else
				{
					dir += 1;
				}
			}
			else
			{
				//right
				if (dir == 0)
				{
					dir = 3;
				}
				else
				{
					dir -= 1;
				}
			}

			//movePred(1) means go straight = do nothing
		}
		else if (movePred(2) > movePred(1))
		{
			//right
			if (dir == 0)
			{
				dir = 3;
			}
			else
			{
				dir -= 1;
			}
		}

		if (sol.fitness < 0)
		{
			sol.fitness = 0;
		}
	}
	window.setFramerateLimit(0);
}


int popSize = 2;
int generations = 1000000;

int mRate = 15;

std::random_device dev;
std::mt19937 seed(dev());

std::uniform_real_distribution<> backprop(-1, 1);
std::uniform_int_distribution<> mutate(0, 100);

void initSolutions(std::vector<GNN::Solution> &solutions)
{
	//random weights for network
	for (int i = 0; i < popSize; i++)
	{
		solutions.push_back(GNN::Solution() = { Eigen::Matrix<float, 5, 32>().setRandom(), Eigen::Matrix<float, 32, 64>().setRandom(), Eigen::Matrix<float, 64, 32>().setRandom(), Eigen::Matrix<float, 32, 16>().setRandom() , Eigen::Matrix<float, 16, 3>().setRandom() });
	}
}

int main()
{
	Snake snek;
	//window.setFramerateLimit(60);
	window.setFramerateLimit(120);

	sf::Event eve;
	srand((unsigned int)time(0));

	foodRect.setPosition(food);
	foodRect.setFillColor(sf::Color::Red);

	bool drawing = false;
	bool playing = false;

	float bestFit = 0;
	GNN::Solution bestSol;

	std::vector<GNN::Solution> solutions;
	std::vector<GNN::Solution> children;

	initSolutions(solutions);
	

	for(int g = 0; g < generations; g++)
	{
		//looping through solutions
		for (int s = 0; s < popSize; s++)
		{
			//resetting solutions fitness
			solutions[s].fitness = 0;

			while (!snek.dead)
			{
				if (drawing)
				{
					window.pollEvent(eve);
					window.clear();

					//SPLIT DRAW SNAKE METHOD
					drawSnake(snek);
					window.draw(foodRect);

					window.display();
				}

				window.pollEvent(eve);

				if (eve.type == sf::Event::KeyPressed)
				{
					if (eve.key.code == sf::Keyboard::Space)
					{
						drawing = !drawing;
					}
				}


				//base game, collison and logic
				gameLogic(snek, solutions[s]);

				snek.time += 1;

				if (colliding(snek, solutions[s]))
				{
					snek.dead = true;
				}

				if (snek.energy <= 0)
				{
					snek.dead = true;
				}

				snek.energy -= 1;


				if (playing)
				{
					window.pollEvent(eve);

					if (eve.type == sf::Event::KeyPressed)
					{
						if (eve.key.code == sf::Keyboard::Right && dir != 0)
						{
							//right
							dir = 2;
						}
						else if (eve.key.code == sf::Keyboard::Left && dir != 2)
						{
							//left
							dir = 0;
						}

						if (eve.key.code == sf::Keyboard::Down && dir != 3)
						{
							//down
							dir = 1;
						}
						else if (eve.key.code == sf::Keyboard::Up && dir != 1)
						{
							//up
							dir = 3;
						}
					}
				}

				//AI Mode
				else
				{
					int foodLeft = 0, foodStraight = 0, foodRight = 0;
					int dis;
					float xDis = food.x - snek.parts[0].x;
					float yDis = food.y - snek.parts[0].y;


					Eigen::Matrix<float, 1, 5> inputvals;

					//left
					if (dir == 0)
					{
						dis = snek.parts[0].x;

						//left
						if (eatFood(sf::Vector2f(snek.parts[0].x, food.y)) && snek.parts[0].y < food.y)
						{
							foodLeft = 1;
						}
						//straight
						else if (eatFood(sf::Vector2f(food.x, snek.parts[0].y)) && snek.parts[0].x > food.x)
						{
							foodStraight = 1;
						}
						//right
						else if (eatFood(sf::Vector2f(snek.parts[0].x, food.y)) && snek.parts[0].y > food.y)
						{
							foodRight = 1;
						}
					}
					//right
					else if(dir == 2)
					{
						dis = 780 - snek.parts[0].x;
						//left
						if (eatFood(sf::Vector2f(snek.parts[0].x, food.y)) && snek.parts[0].y > food.y)
						{
 							foodLeft = 1;
						}
						//straight
						else if (eatFood(sf::Vector2f(food.x, snek.parts[0].y)) && snek.parts[0].x < food.x)
						{
							foodStraight = 1;
						}
						//right
						else if (eatFood(sf::Vector2f(snek.parts[0].x, food.y)) && snek.parts[0].y < food.y)
						{
							foodRight = 1;
						}
					}
					//up
					else if(dir == 3)
					{
						dis = snek.parts[0].y;

						//left
						if (eatFood(sf::Vector2f(food.x, snek.parts[0].y)) && snek.parts[0].x > food.x)
						{
							foodLeft = 1;
						}
						//straight
						else if (eatFood(sf::Vector2f(snek.parts[0].x, food.y)) && snek.parts[0].y > food.y)
						{
							foodStraight = 1;
						}
						//right
						else if (eatFood(sf::Vector2f(food.x, snek.parts[0].y)) && snek.parts[0].x < food.x)
						{
							foodRight = 1;
						}
					}
					//down
					else if (dir == 1)
					{
						dis = 580 - snek.parts[0].y;

						//left
						if (eatFood(sf::Vector2f(food.x, snek.parts[0].y)) && snek.parts[0].x > food.x)
						{
							foodLeft = 1;
						}
						//straight
						else if (eatFood(sf::Vector2f(snek.parts[0].x, food.y)) && snek.parts[0].y < food.y)
						{
							foodStraight = 1;
						}
						//right
						else if (eatFood(sf::Vector2f(food.x, snek.parts[0].y)) && snek.parts[0].x < food.x)
						{
							foodRight = 1;
						}
					}

					inputvals << foodLeft, foodStraight, foodRight, dir + 1, dis;

					//Movement Prediction
					Eigen::Vector3f movePred = GNN::feed(solutions[s], inputvals);

					if (movePred(0) > movePred(1))
					{
						if (movePred(0) > movePred(2))
						{
							//left
							if (dir == 3)
							{
								dir = 0;
							}
							else
							{
								dir += 1;
							}
						}
						else
						{
							//right
							if (dir == 0)
							{
								dir = 3;
							}
							else
							{
								dir -= 1;
							}
						}

						//movePred(1) means go straight = do nothing
					}
					else if (movePred(2) > movePred(1))
					{
						//right
						if (dir == 0)
						{
							dir = 3;
						}
						else
						{
							dir -= 1;
						}
					}

					
					//adding fitness for correct prediction
					
					if (dir == 0 && signbit(xDis))
					{
						solutions[s].fitness += 0.5;
					}
					else if (dir == 3 && signbit(yDis))
					{
						solutions[s].fitness += 0.5;
					}
					else if (dir == 2 && !signbit(xDis))
					{
						solutions[s].fitness += 0.5;
					}
					else if (dir == 1 && !signbit(yDis))
					{
						solutions[s].fitness += 0.5;
					}
					else
					{
						solutions[s].fitness -= 1;
					}
					
					
					
				}

				//if (solutions[s].fitness < 0)
				//{
				//	solutions[s].fitness = 0;
				//}
			}

			//After Snake Dies

			solutions[s].fitness += (snek.score * 25) + (snek.time * 0.1) - 3;

			if (solutions[s].fitness < 0)
			{
				solutions[s].fitness = 0;
			}

			//Resetting variables
			snek = Snake();
			food = sf::Vector2f(rand() % 700 + 100, rand() % 500 + 100);

			//if (food.y <= 350 && food.y >= 250)
			//{
			//	food = sf::Vector2f(rand() % 700 + 100, rand() % 500 + 100);
			//}

			foodRect.setPosition(food);
		}

		//sorting by fitness
		std::sort(solutions.begin(), solutions.end(), [](const GNN::Solution& s1, const GNN::Solution& s2)
			{
				return s1.fitness > s2.fitness;
			});

		
		if (solutions[0].fitness >= bestFit)
		{
			bestFit = solutions[0].fitness;
			bestSol = solutions[0];
			std::cout << bestFit << "\n";
		}

		//if (g % 4 == 0)
		//{
			//playSolution(solutions[0]);
		//}

		int test;

		if (popSize <= 10)
		{
			children.push_back(solutions[0]);
			test = popSize - 1;
		}
		else
		{
			//elitism
			for (int i = 0; i < popSize * 0.10; i++)
			{
				children.push_back(solutions[i]);
			}
			test = (popSize - popSize * 0.10) / 2;
		}


		//crossover / selection / mutation
		for (int i = 0; i < test; i++)
		{
			//selection
			int par1 = roulette(solutions);
			int par2 = roulette(solutions);

			if (par1 == -1 || par2 == -1)
			{
				throw std::exception("bad input");
				return -1;
			}

			//crossover
			Eigen::MatrixXf c1w1 = GNN::crossover(solutions[par1].w1, solutions[par2].w1);
			Eigen::MatrixXf c1w2 = GNN::crossover(solutions[par1].w2, solutions[par2].w2);
			Eigen::MatrixXf c1w3 = GNN::crossover(solutions[par1].w3, solutions[par2].w3);
			Eigen::MatrixXf c1w4 = GNN::crossover(solutions[par1].w4, solutions[par2].w4);
			Eigen::MatrixXf c1w5 = GNN::crossover(solutions[par1].w5, solutions[par2].w5);

			Eigen::MatrixXf c2w1 = GNN::crossover(solutions[par2].w1, solutions[par1].w1);
			Eigen::MatrixXf c2w2 = GNN::crossover(solutions[par2].w2, solutions[par1].w2);
			Eigen::MatrixXf c2w3 = GNN::crossover(solutions[par2].w3, solutions[par1].w3);
			Eigen::MatrixXf c2w4 = GNN::crossover(solutions[par2].w4, solutions[par1].w4);
			Eigen::MatrixXf c2w5 = GNN::crossover(solutions[par2].w5, solutions[par1].w5);

			GNN::Solution child = { c1w1, c1w2, c1w3, c1w4, c1w5 };
			GNN::Solution child2 = { c2w1, c2w2, c2w3, c2w4, c2w5 };

			//mutation
			for (int j = 0; j < child.w1.size(); j++)
			{
				//backprop / adjustments
				child.w1(j) += backprop(seed);
				child2.w1(j) += backprop(seed);

				if (mutate(seed) < mRate)
				{
					//CHANGETHIS
					child.w1(j) = 0;
				}
				if (mutate(seed) < mRate)
				{
					//CHANGETHIS
					child2.w1(j) = 0;
				}
			}
			for (int j = 0; j < child.w2.size(); j++)
			{
				child.w2(j) += backprop(seed);
				child2.w2(j) += backprop(seed);

				if (mutate(seed) < mRate)
				{
					//CHANGETHIS
					child.w2(j) = 0;
				}
				if (mutate(seed) < mRate)
				{
					//CHANGETHIS
					child2.w2(j) = 0;
				}
			}
			for (int j = 0; j < child.w3.size(); j++)
			{
				child.w3(j) += backprop(seed);
				child2.w3(j) += backprop(seed);

				if (mutate(seed) < mRate)
				{
					//CHANGETHIS
					child.w3(j) = 0;
				}
				if (mutate(seed) < mRate)
				{
					//CHANGETHIS
					child2.w3(j) = 0;
				}
			}
			for (int j = 0; j < child.w4.size(); j++)
			{
				//backprop / adjustments
				child.w4(j) += backprop(seed);
				child2.w4(j) += backprop(seed);

				if (mutate(seed) < mRate)
				{
					//CHANGETHIS
					child.w4(j) = 0;
				}
				if (mutate(seed) < mRate)
				{
					//CHANGETHIS
					child2.w4(j) = 0;
				}
			}
			for (int j = 0; j < child.w5.size(); j++)
			{
				//backprop / adjustments
				child.w5(j) += backprop(seed);
				child2.w5(j) += backprop(seed);

				if (mutate(seed) < mRate)
				{
					//CHANGETHIS
					child.w5(j) = 0;
				}
				if (mutate(seed) < mRate)
				{
					//CHANGETHIS
					child2.w5(j) = 0;
				}
			}

			children.push_back(child);
			children.push_back(child2);
		}

		if (children.size() != popSize)
		{
			for (int c = children.size(); c < popSize; c++)
			{
				children.push_back(GNN::Solution() = { Eigen::Matrix<float, 5, 32>().setRandom(), Eigen::Matrix<float, 32, 64>().setRandom(), Eigen::Matrix<float, 64, 32>().setRandom(), Eigen::Matrix<float, 32, 16>().setRandom() , Eigen::Matrix<float, 16, 3>().setRandom() });
			}
		}

		solutions = children;
		children.clear();
	}

	return 0;
}

