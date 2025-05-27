"""
AI algorithms for Tetris.
"""
import pygame
import random
import numpy as np
import time
import threading
import matplotlib.pyplot as plt
from constants import *
from utils import plot_to_surface

# Try to import torch for neural network functionality
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Neural network features will be disabled.")

class TetrisNN(nn.Module):
    """Neural network model for Tetris board evaluation"""
    def __init__(self, input_size=NN_INPUT_SIZE, hidden_size=NN_HIDDEN_SIZE, output_size=NN_OUTPUT_SIZE):
        super(TetrisNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class GeneticOptimizer:
    """Genetic algorithm for optimizing AI weights"""
    def __init__(self, population_size=GA_POPULATION_SIZE, generations=GA_GENERATIONS, mutation_rate=GA_MUTATION_RATE):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        # Weight ranges for initialization
        self.weight_ranges = {
            'holes': (-10.0, -0.1),
            'bumpiness': (-5.0, -0.1),
            'height': (-5.0, -0.1),
            'lines_cleared': (0.1, 10.0),
            'wells': (-5.0, -0.1),
            'row_transitions': (-5.0, -0.1),
            'col_transitions': (-5.0, -0.1)
        }
        
        # Statistics
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.current_generation = 0
        self.running = False
    
    def create_individual(self):
        """Create a random individual (set of weights)"""
        return {
            key: random.uniform(min_val, max_val)
            for key, (min_val, max_val) in self.weight_ranges.items()
        }
    
    def create_population(self):
        """Create an initial population of random individuals"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def evaluate_fitness(self, weights, tetris_game, num_games=3, max_moves=200):
        """Evaluate the fitness of a set of weights by playing games"""
        # Save original weights
        original_weights = tetris_game.ai_weights.copy()
        
        # Set new weights
        tetris_game.ai_weights = weights
        
        total_score = 0
        total_lines = 0
        
        for _ in range(num_games):
            tetris_game.reset()
            moves = 0
            
            while not tetris_game.game_over and moves < max_moves:
                tetris_game.apply_ai_move()
                moves += 1
            
            total_score += tetris_game.score
            total_lines += tetris_game.lines_cleared
        
        # Restore original weights
        tetris_game.ai_weights = original_weights
        
        # Fitness is a combination of score and lines cleared
        return (total_score / num_games) + (total_lines / num_games) * 100
    
    def select_parents(self, population, fitnesses):
        """Select parents for reproduction using tournament selection"""
        tournament_size = GA_TOURNAMENT_SIZE
        
        def select_one():
            # Randomly select individuals for the tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            
            # Select the best individual from the tournament
            winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
            return population[winner_idx]
        
        return select_one(), select_one()
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents to create a child"""
        child = {}
        
        for key in parent1.keys():
            # Uniform crossover: randomly select from either parent
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        
        return child
    
    def mutate(self, individual):
        """Apply mutation to an individual"""
        mutated = individual.copy()
        
        for key in mutated.keys():
            # Apply mutation with probability mutation_rate
            if random.random() < self.mutation_rate:
                min_val, max_val = self.weight_ranges[key]
                # Add a random value within ±20% of the range
                range_size = max_val - min_val
                mutation_amount = random.uniform(-0.2 * range_size, 0.2 * range_size)
                mutated[key] += mutation_amount
                
                # Ensure the value stays within the valid range
                mutated[key] = max(min_val, min(max_val, mutated[key]))
        
        return mutated
    
    def train(self, tetris_game, callback=None):
        """
        Train the AI using genetic algorithm
        
        Args:
            tetris_game: Tetris game instance to use for evaluation
            callback: Optional function to call after each generation with progress info
            
        Returns:
            dict: Best weights found
        """
        self.running = True
        self.current_generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # Create initial population
        population = self.create_population()
        
        # Track the best individual across all generations
        self.best_individual = None
        self.best_fitness = float('-inf')
        
        for generation in range(self.generations):
            if not self.running:
                break
                
            self.current_generation = generation + 1
            
            # Evaluate fitness for each individual
            fitnesses = []
            for individual in population:
                fitness = self.evaluate_fitness(individual, tetris_game)
                fitnesses.append(fitness)
                
                # Update best individual if needed
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = individual.copy()
            
            # Store statistics
            self.best_fitness_history.append(max(fitnesses))
            self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
            
            # Call callback if provided
            if callback:
                progress = (generation + 1) / self.generations
                callback(progress, self.best_individual, self.best_fitness)
            
            # Create the next generation
            new_population = []
            
            # Elitism: keep the best individual
            if GA_ELITISM:
                best_idx = fitnesses.index(max(fitnesses))
                new_population.append(population[best_idx])
            
            # Create the rest of the population through selection, crossover, and mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self.select_parents(population, fitnesses)
                
                # Create child through crossover
                child = self.crossover(parent1, parent2)
                
                # Apply mutation
                child = self.mutate(child)
                
                # Add to new population
                new_population.append(child)
            
            # Update population
            population = new_population
        
        self.running = False
        return self.best_individual
    
    def stop_training(self):
        """Stop the training process"""
        self.running = False
    
    def get_training_plot(self, width, height):
        """Get a plot of the training progress"""
        if not self.best_fitness_history:
            return None
            
        plt.figure(figsize=(8, 4))
        plt.plot(self.best_fitness_history, label='Best Fitness')
        plt.plot(self.avg_fitness_history, label='Average Fitness')
        plt.title('Genetic Algorithm Training Progress')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        
        plot_surface = plot_to_surface(plt.gcf(), width, height)
        plt.close()
        
        return plot_surface

class NeuralNetworkTrainer:
    """Trainer for the neural network model"""
    def __init__(self, input_size=NN_INPUT_SIZE, hidden_size=NN_HIDDEN_SIZE, output_size=NN_OUTPUT_SIZE):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural network training")
        
        self.model = TetrisNN(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=NN_LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
        # Training data
        self.states = []
        self.targets = []
        
        # Statistics
        self.loss_history = []
        self.running = False
        self.current_epoch = 0
        self.total_epochs = 0
    
    def collect_training_data(self, tetris_game, num_games=10, max_moves=200, callback=None):
        """Collect training data by playing games with the current best heuristic"""
        self.running = True
        self.states = []
        self.targets = []
        
        for game_idx in range(num_games):
            if not self.running:
                break
                
            tetris_game.reset()
            moves = 0
            
            while not tetris_game.game_over and moves < max_moves and self.running:
                # Find best move using heuristic
                best_move = tetris_game.find_best_move()
                if not best_move:
                    break
                
                # Extract features before making the move
                features = tetris_game.get_board_features()
                feature_vector = [
                    features['aggregate_height'] / (tetris_game.height * tetris_game.width),
                    features['bumpiness'] / tetris_game.width,
                    features['holes'] / (tetris_game.height * tetris_game.width),
                    features['wells'] / (tetris_game.height * tetris_game.width),
                    features['row_transitions'] / (tetris_game.height * tetris_game.width),
                    features['col_transitions'] / (tetris_game.height * tetris_game.width),
                    features['complete_lines'] / tetris_game.height,
                    tetris_game.level / 20  # Normalize level
                ]
                
                # Apply the move
                tetris_game.apply_ai_move()
                
                # Use the heuristic evaluation as the target
                target = best_move['evaluation']
                
                # Store the data
                self.states.append(feature_vector)
                self.targets.append(target)
                
                moves += 1
            
            # Call callback if provided
            if callback:
                progress = (game_idx + 1) / num_games
                callback(progress, len(self.states), 0)
        
        self.running = False
        return len(self.states)
    
    def train(self, epochs=100, batch_size=32, callback=None):
        """Train the neural network on collected data"""
        if not self.states:
            raise ValueError("No training data available. Call collect_training_data first.")
        
        self.running = True
        self.current_epoch = 0
        self.total_epochs = epochs
        self.loss_history = []
        
        # Convert data to tensors
        X = torch.FloatTensor(self.states)
        y = torch.FloatTensor(self.targets).unsqueeze(1)
        
        # Training loop
        for epoch in range(epochs):
            if not self.running:
                break
                
            self.current_epoch = epoch + 1
            
            # Shuffle data
            indices = torch.randperm(len(self.states))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(self.states), batch_size):
                # Get batch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss
            avg_loss = total_loss / num_batches
            self.loss_history.append(avg_loss)
            
            # Call callback if provided
            if callback:
                progress = (epoch + 1) / epochs
                callback(progress, len(self.states), avg_loss)
        
        # Save the trained model
        if self.running:
            torch.save(self.model.state_dict(), NN_MODEL_PATH)
        
        self.running = False
        return self.model
    
    def stop_training(self):
        """Stop the training process"""
        self.running = False
    
    def get_training_plot(self, width, height):
        """Get a plot of the training progress"""
        if not self.loss_history:
            return None
            
        plt.figure(figsize=(8, 4))
        plt.plot(self.loss_history)
        plt.title('Neural Network Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plot_surface = plot_to_surface(plt.gcf(), width, height)
        plt.close()
        
        return plot_surface

def load_nn_model():
    """Load the neural network model if available"""
    if not TORCH_AVAILABLE:
        return None
        
    try:
        model = TetrisNN()
        model.load_state_dict(torch.load(NN_MODEL_PATH))
        model.eval()
        print("Neural network model loaded successfully")
        return model
    except:
        print("No pre-trained neural network model found")
        return None

def evaluate_with_nn(model, features, tetris_game):
    """Evaluate a board state using the neural network"""
    if not model or not TORCH_AVAILABLE:
        return None
        
    try:
        feature_vector = [
            features['aggregate_height'] / (tetris_game.height * tetris_game.width),
            features['bumpiness'] / tetris_game.width,
            features['holes'] / (tetris_game.height * tetris_game.width),
            features['wells'] / (tetris_game.height * tetris_game.width),
            features['row_transitions'] / (tetris_game.height * tetris_game.width),
            features['col_transitions'] / (tetris_game.height * tetris_game.width),
            features['complete_lines'] / tetris_game.height,
            tetris_game.level / 20  # Normalize level
        ]
        
        input_tensor = torch.FloatTensor(feature_vector)
        with torch.no_grad():
            score = model(input_tensor).item()
        return score
    except Exception as e:
        print(f"Neural network evaluation error: {e}")
        return None

def start_genetic_training(tetris_game, callback=None):
    """Start genetic algorithm training in a separate thread"""
    optimizer = GeneticOptimizer()
    
    def training_thread():
        best_weights = optimizer.train(tetris_game, callback)
        
        # Apply the best weights
        if best_weights:
            tetris_game.ai_weights = best_weights
            print("Genetic training complete. Best weights:", best_weights)
    
    thread = threading.Thread(target=training_thread)
    thread.daemon = True
    thread.start()
    
    return optimizer

def start_nn_training(tetris_game, callback=None):
    """Start neural network training in a separate thread"""
    if not TORCH_AVAILABLE:
        return None
    
    trainer = NeuralNetworkTrainer()
    
    def training_thread():
        # Collect training data
        print("Collecting training data...")
        trainer.collect_training_data(tetris_game, callback=callback)
        
        # Train the model
        print("Training neural network...")
        model = trainer.train(callback=callback)
        
        # Apply the trained model
        if model:
            print("Neural network training complete.")
    
    thread = threading.Thread(target=training_thread)
    thread.daemon = True
    thread.start()
    
    return trainer
