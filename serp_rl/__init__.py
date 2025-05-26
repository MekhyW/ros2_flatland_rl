#!/usr/bin/env python3
import rclpy
from rclpy.publisher import Publisher
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from sensor_msgs.msg import LaserScan
from flatland_msgs.srv import MoveModel
from flatland_msgs.msg import Collisions

from gym import Env
from gym.spaces import Discrete, Box

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import numpy as np
import time
import threading
import matplotlib.pyplot as plt
import json
from datetime import datetime

class SerpControllerEnv(Node, Env):
    def __init__(self) -> None:
        super().__init__("SerpControllerEnv")

        # Predefined speed for the robot
        linear_speed = 0.5
        angular_speed = 1.57079632679

        # Set of actions. Defined by their linear and angular speed
        self.actions = [(linear_speed, 0.0), # move forward
                        (0.0, angular_speed), # rotate left
                        (0.0, -angular_speed)] # rotate right

        # How close the robot needs to be to the target to finish the task
        self.end_range = 0.2

        # Number of divisions of the LiDAR
        self.n_lidar_sections = 9
        self.lidar_sample = []

        # Variables that track a possible end state
        # current distance to target
        self.distance_to_end = 10.0
        # true if a collision happens
        self.collision = False

        # Possible starting positions
        # Updated positions for room_map (multi-room environment)
        self.start_positions = [(-8.0, -4.0, 0.0), (8.0, 4.0, 3.14159265359)]
        # Current position
        self.position = 0

        self.step_number = 0

        # Maximum number of steps before it times out
        self.max_steps = 500

        # Records previous action taken. At the start of an episode, there is no prior action so -1 is assigned
        self.previous_action = -1

        # Used for data collection during training
        self.total_step_cnt = 0
        self.total_episode_cnt = 0
        self.training = False
        
        # Initialize state with default values
        self.state = np.ones(self.n_lidar_sections) * 0.5  # Middle range default
                                    
        # **** Create publishers ****
        self.pub:Publisher = self.create_publisher(Twist, "/cmd_vel", 1)
        # ***************************

        # **** Create subscriptions ****
        self.create_subscription(LaserScan, "/static_laser", self.processLiDAR, 1)

        self.create_subscription(LaserScan, "/end_beacon_laser", self.processEndLiDAR, 1)

        self.create_subscription(Collisions, "/collisions", self.processCollisions, 1)
        # ******************************

        # **** Define action and state spaces ****

        # action is an integer between 0 and 2 (total of 3 actions)
        self.action_space = Discrete(len(self.actions))
        # state is represented by a numpy.Array with size 9 and values between 0 and 1
        self.observation_space = Box(0, 1, shape=(self.n_lidar_sections,), dtype=np.float32)

        # ****************************************

    # Resets the environment to an initial state
    def reset(self):
        # Make sure the robot is stopped
        self.change_robot_speeds(self.pub, 0.0, 0.0)

        if self.total_step_cnt != 0: self.total_episode_cnt += 1

        # **** Move robot and end beacon to new positions ****
        start_pos = self.start_positions[self.position]
        self.position = 1 - self.position
        end_pos = self.start_positions[self.position]
        
        self.move_model('serp', start_pos[0], start_pos[1], start_pos[2])
        self.move_model('end_beacon', end_pos[0], end_pos[1], 0.0)
        # ****************************************************

        # **** Reset necessary values ****
        self.lidar_sample = []
        
        # Wait for valid lidar reading with timeout
        start_time = time.time()
        while len(self.lidar_sample) != self.n_lidar_sections:
            if time.time() - start_time > 5.0:  # 5 second timeout
                self.get_logger().warn("Timeout waiting for lidar data in reset. Using default values.")
                self.lidar_sample = [0.5] * self.n_lidar_sections
                break
            time.sleep(0.01)
        
        self.state = np.array(self.lidar_sample, dtype=np.float32)
        
        # Validate state
        if np.any(np.isnan(self.state)) or np.any(np.isinf(self.state)):
            self.get_logger().warn("Invalid state detected in reset. Using default values.")
            self.state = np.ones(self.n_lidar_sections, dtype=np.float32) * 0.5

        # Flatland can sometimes send several collision messages. 
        # This makes sure that no collisions are wrongfully detected at the start of an episode 
        time.sleep(0.1)

        self.distance_to_end = 10.0
        self.collision = False
        self.step_number = 0
        self.previous_action = -1
        # ********************************

        return self.state

    # Performs a step for the RL agent
    def step(self, action): 

        # **** Performs the action and waits for it to be completed ****
        self.change_robot_speeds(self.pub, self.actions[action][0], self.actions[action][1])

        self.lidar_sample = []
        
        # Wait for lidar with timeout
        start_time = time.time()
        while len(self.lidar_sample) != self.n_lidar_sections:
            if time.time() - start_time > 2.0:  # 2 second timeout
                self.get_logger().warn("Timeout waiting for lidar data in step. Using previous state.")
                self.lidar_sample = self.state.tolist()
                break
            time.sleep(0.01)
            
        self.change_robot_speeds(self.pub, 0.0, 0.0)
        # **************************************************************

        # Register current state
        self.state = np.array(self.lidar_sample, dtype=np.float32)
        
        # Validate state
        if np.any(np.isnan(self.state)) or np.any(np.isinf(self.state)):
            self.get_logger().warn("Invalid state detected in step. Using previous valid state.")
            # Use previous state or default
            if hasattr(self, '_last_valid_state'):
                self.state = self._last_valid_state.copy()
            else:
                self.state = np.ones(self.n_lidar_sections, dtype=np.float32) * 0.5
        else:
            self._last_valid_state = self.state.copy()

        self.step_number += 1
        self.total_step_cnt += 1

        # **** Calculates the reward and determines if an end state was reached ****
        done = False

        end_state = ''

        if self.collision:
            end_state = "collision"
            reward = -200
            done = True
        elif self.distance_to_end < self.end_range:
            end_state = "finished"
            reward = 400 + (200 - self.step_number)
            done = True
        elif self.step_number >= self.max_steps:
            end_state = "timeout"
            reward = -300 
            done = True
        elif action == 0:
            reward = 2
        else:
            reward = 0
        # **************************************************************************

        info = {'end_state': end_state}

        if done and self.training:
            self.get_logger().info('Training - Episode ' + str(self.total_episode_cnt) + ' end state: ' + end_state)
            self.get_logger().info('Total steps: ' + str(self.total_step_cnt))

        return self.state, reward, done, info

    def render(self): pass

    def close(self): pass

    def reset_counters(self):
        self.total_step_cnt = 0
        self.total_episode_cnt = 0

    # Change the speed of the robot
    def change_robot_speeds(self, publisher, linear, angular):
        twist_msg = Twist()
        twist_msg.linear.x = linear
        twist_msg.angular.z = angular
        publisher.publish(twist_msg)

    # Waits for a new LiDAR reading.
    # A new LiDAR reading is also used to signal when an action should finish being performed.
    def wait_lidar_reading(self):
        start_time = time.time()
        while len(self.lidar_sample) != self.n_lidar_sections:
            if time.time() - start_time > 5.0:
                self.get_logger().warn("Timeout in wait_lidar_reading")
                self.lidar_sample = [0.5] * self.n_lidar_sections
                break
            time.sleep(0.01)

    # Send a request to move a model
    def move_model(self, model_name, x, y, theta):
        self.get_logger().info(f'Moving {model_name} to x={x}, y={y}, theta={theta}')
        client = self.create_client(MoveModel, "/move_model")
        client.wait_for_service()
        request = MoveModel.Request()
        request.name = model_name
        request.pose = Pose2D()
        request.pose.x = x
        request.pose.y = y
        request.pose.theta = theta
        client.call_async(request)
    
    # Sample LiDAR data
    # Divite into sections and sample the lowest value from each
    def processLiDAR(self, data):
        try:
            self.lidar_sample = []
            max_range = 10.0  # Maximum LiDAR range
            
            rays = data.ranges
            rays_per_section = len(rays) // self.n_lidar_sections

            for i in range(self.n_lidar_sections - 1):
                section_rays = rays[rays_per_section * i:rays_per_section * (i + 1)]
                # Filter out nan and inf values
                valid_rays = [r for r in section_rays if not (np.isnan(r) or np.isinf(r))]
                
                if valid_rays:
                    min_val = min(valid_rays)
                else:
                    # If no valid rays, use max range
                    min_val = max_range
                
                # Normalize to 0-1 range
                normalized_val = min(min_val / max_range, 1.0)
                self.lidar_sample.append(normalized_val)
            
            # Last section
            last_section = rays[(self.n_lidar_sections - 1) * rays_per_section:]
            valid_last = [r for r in last_section if not (np.isnan(r) or np.isinf(r))]
            
            if valid_last:
                last_min = min(valid_last)
            else:
                last_min = max_range
                
            self.lidar_sample.append(min(last_min / max_range, 1.0))
            
            # Final validation
            for i in range(len(self.lidar_sample)):
                if np.isnan(self.lidar_sample[i]) or np.isinf(self.lidar_sample[i]):
                    self.lidar_sample[i] = 1.0  # Max normalized range
                    
        except Exception as e:
            self.get_logger().error(f"Error in processLiDAR: {str(e)}")
            self.lidar_sample = [0.5] * self.n_lidar_sections

    
    # Handle end beacon LiDAR data
    # Lowest value is the distance from robot to target
    def processEndLiDAR(self, data):
        try:
            clean_data = [x for x in data.ranges if not (np.isnan(x) or np.isinf(x)) and x > 0]
            if clean_data:
                self.distance_to_end = min(clean_data)
            else:
                self.distance_to_end = 10.0  # Default to far distance
        except Exception as e:
            self.get_logger().error(f"Error in processEndLiDAR: {str(e)}")
            self.distance_to_end = 10.0
    
    # Process collisions
    def processCollisions(self, data):
        if len(data.collisions) > 0:
            self.collision = True

    # Run an entire episode manually for testing purposes
    # return true if succesful
    def run_episode(self, agent):
        
        com_reward = 0

        obs = self.reset()
        done = False
        while not done:
            action, states = agent.predict(obs, deterministic=True)
            obs, rewards, done, info = self.step(action)
            com_reward += rewards
        
        self.get_logger().info('Episode concluded. End state: ' + info['end_state'] + '  Commulative reward: ' + str(com_reward))

        return info['end_state'] == 'finished'

    def train_single_hyperparameter_set(self, name, hyperparams, episodes_per_eval=50, eval_episodes=10, max_episodes=500):
        """Train a single hyperparameter set and track results"""
        
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info(f"Testing hyperparameter set: {name}")
        self.get_logger().info(f"{'='*60}")
        
        # Set seeds for reproducibility
        import random
        import numpy as np
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        
        # Create agent with current hyperparameters
        agent = PPO(
            "MlpPolicy", 
            self, 
            verbose=1, 
            seed=seed,
            **hyperparams
        )
        
        # Track results
        results = {
            'name': name,
            'episodes': [],
            'accuracies': [],
            'avg_rewards': [],
            'training_time': []
        }
        
        episode_count = 0
        best_accuracy = 0
        start_time = time.time()
        
        while episode_count < max_episodes:
            # Training phase
            episodes_to_train = min(episodes_per_eval, max_episodes - episode_count)
            
            self.get_logger().info(f'\n[{name}] Training episodes {episode_count} to {episode_count + episodes_to_train}')
            
            self.training = True
            self.reset_counters()
            
            # Calculate steps needed
            steps_per_episode = 50
            training_steps = episodes_to_train * steps_per_episode
            
            # Train with error handling
            try:
                agent.learn(total_timesteps=training_steps, reset_num_timesteps=False)
            except Exception as e:
                self.get_logger().error(f"Error during training: {str(e)}")
                self.get_logger().info("Attempting to recover and continue training...")
                # Reset to a known good state
                self.reset()
                time.sleep(1)
                continue
            
            episode_count += episodes_to_train
            self.training = False
            
            # Evaluation phase
            self.get_logger().info(f'[{name}] Evaluating after {episode_count} episodes...')
            successful_episodes = 0
            total_rewards = 0
            
            for i in range(eval_episodes):
                obs = self.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, done, info = self.step(action)
                    episode_reward += reward
                
                total_rewards += episode_reward
                if info['end_state'] == 'finished':
                    successful_episodes += 1
            
            accuracy = successful_episodes / eval_episodes
            avg_reward = total_rewards / eval_episodes
            current_time = time.time() - start_time
            
            # Store results
            results['episodes'].append(episode_count)
            results['accuracies'].append(accuracy)
            results['avg_rewards'].append(avg_reward)
            results['training_time'].append(current_time)
            
            self.get_logger().info(f'[{name}] Episodes: {episode_count}, Accuracy: {accuracy:.2%}, Avg Reward: {avg_reward:.2f}')
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                agent.save(f"ppo_best_{name}")
            
            # Early stopping if target reached
            if accuracy >= 0.8:
                self.get_logger().info(f'[{name}] Target accuracy reached! Training complete.')
                break
        
        # Final save
        agent.save(f"ppo_final_{name}")
        results['best_accuracy'] = best_accuracy
        results['final_time'] = time.time() - start_time
        
        return results

    def plot_comparison(self, all_results):
        """Create comparison plots for all hyperparameter sets"""
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hyperparameter Comparison', fontsize=16)
        
        # Define colors for each set
        colors = ['blue', 'red', 'green']
        
        # Plot 1: Accuracy over episodes
        ax1.set_title('Accuracy vs Episodes')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Accuracy')
        for i, result in enumerate(all_results):
            ax1.plot(result['episodes'], result['accuracies'], 
                    color=colors[i], label=result['name'], linewidth=2)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # Plot 2: Average reward over episodes
        ax2.set_title('Average Reward vs Episodes')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Average Reward')
        for i, result in enumerate(all_results):
            ax2.plot(result['episodes'], result['avg_rewards'], 
                    color=colors[i], label=result['name'], linewidth=2)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy over time
        ax3.set_title('Accuracy vs Training Time')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Accuracy')
        for i, result in enumerate(all_results):
            ax3.plot(result['training_time'], result['accuracies'], 
                    color=colors[i], label=result['name'], linewidth=2)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.05)
        
        # Plot 4: Final comparison bar chart
        ax4.set_title('Final Performance Comparison')
        names = [r['name'] for r in all_results]
        best_accuracies = [r['best_accuracy'] for r in all_results]
        final_times = [r['final_time'] for r in all_results]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, best_accuracies, width, label='Best Accuracy', color='skyblue')
        ax4.set_ylabel('Best Accuracy', color='blue')
        ax4.tick_params(axis='y', labelcolor='blue')
        ax4.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2%}', ha='center', va='bottom')
        
        # Create second y-axis for time
        ax4_2 = ax4.twinx()
        bars2 = ax4_2.bar(x + width/2, final_times, width, label='Training Time (s)', color='lightcoral')
        ax4_2.set_ylabel('Training Time (seconds)', color='red')
        ax4_2.tick_params(axis='y', labelcolor='red')
        
        ax4.set_xlabel('Hyperparameter Set')
        ax4.set_xticks(x)
        ax4.set_xticklabels(names)
        
        # Add legend
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_2.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hyperparameter_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.get_logger().info(f"\nComparison plot saved as: {filename}")
        
        # Also save results as JSON for future reference
        json_filename = f"hyperparameter_results_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        self.get_logger().info(f"Results data saved as: {json_filename}")
        
        plt.show()

    def run_rl_alg(self):
        """Run comparison of different hyperparameter sets"""
        
        check_env(self)
        
        # Give the environment time to initialize
        time.sleep(2)
        self.wait_lidar_reading()
        
        # Define three hyperparameter sets with more conservative values
        hyperparameter_sets = {
            "Original": {
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "normalize_advantage": True,
                "target_kl": 0.01,
            },
            "Aggressive": {
                "n_steps": 256,
                "batch_size": 128,
                "n_epochs": 20,
                "learning_rate": 5e-4,
                "gamma": 0.95,
                "gae_lambda": 0.90,
                "clip_range": 0.3,
                "ent_coef": 0.02,
                "vf_coef": 0.5,
                "max_grad_norm": 0.7,
                "normalize_advantage": True,
                "target_kl": 0.02,
            },
            "Conservative": {
                "n_steps": 4096,
                "batch_size": 32,
                "n_epochs": 5,
                "learning_rate": 1e-4,
                "gamma": 0.995,
                "gae_lambda": 0.98,
                "clip_range": 0.1,
                "ent_coef": 0.001,
                "vf_coef": 1.0,
                "max_grad_norm": 0.3,
                "normalize_advantage": True,
                "target_kl": 0.005,
            }
        }
        
        # Print hyperparameter comparison
        self.get_logger().info("\n" + "="*80)
        self.get_logger().info("HYPERPARAMETER COMPARISON")
        self.get_logger().info("="*80)
        
        for name, params in hyperparameter_sets.items():
            self.get_logger().info(f"\n{name}:")
            for key, value in params.items():
                self.get_logger().info(f"  {key}: {value}")
        
        # Train each hyperparameter set
        all_results = []
        
        for name, hyperparams in hyperparameter_sets.items():
            # Reset environment between runs
            self.reset()
            time.sleep(1)  # Give time for environment to stabilize
            
            # Train and collect results
            results = self.train_single_hyperparameter_set(
                name=name,
                hyperparams=hyperparams,
                episodes_per_eval=50,
                eval_episodes=10,
                max_episodes=200
            )
            
            all_results.append(results)
            
            # Short break between different hyperparameter sets
            time.sleep(2)
        
        # Create comparison plots
        self.plot_comparison(all_results)
        
        # Print final summary
        self.get_logger().info("\n" + "="*80)
        self.get_logger().info("FINAL SUMMARY")
        self.get_logger().info("="*80)
        
        for result in all_results:
            self.get_logger().info(f"\n{result['name']}:")
            self.get_logger().info(f"  Best Accuracy: {result['best_accuracy']:.2%}")
            self.get_logger().info(f"  Training Time: {result['final_time']:.2f} seconds")
            self.get_logger().info(f"  Episodes to 80% accuracy: {next((e for e, a in zip(result['episodes'], result['accuracies']) if a >= 0.8), 'Not reached')}")

def main(args = None):
    rclpy.init()
    
    serp = SerpControllerEnv()

    thread = threading.Thread(target=serp.run_rl_alg)
    thread.start()

    rclpy.spin(serp)



if __name__ == "__main__":
    main()