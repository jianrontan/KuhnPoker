import random
import numpy as np


class KuhnPoker:
    """
    This is the Kuhn Poker implementation
    """

    def __init__(self):
        # J, Q, K
        self.cards = [0, 1, 2]
        # Pass fold / bet
        self.actions = ["pass", "bet"]

    # Deal cards
    def deal_cards(self):
        cards = random.sample(self.cards, 2)
        return cards[0], cards[1]

    # If it is a final node
    def is_terminal(self, history):
        if len(history) > 1:
            if history[-2:] == "pp" or history[-2:] == "bp" or history[-2:] == "bb":
                return True
        return False

    def get_utility(self, history, p0_card, p1_card):
        # Player who starts is p0
        if history == "pp":
            return 1 if p0_card > p1_card else -1
        elif history == "bp":
            return 1
        elif history == "bb":
            return 2 if p0_card > p1_card else -2
        elif history == "pbb":
            return 2 if p0_card > p1_card else -2
        elif history == "pbp":
            return -1
        else:
            raise ValueError(
                f"Invalid history for utility calculation: {history}")


class InformationSet:
    """
    Each infoset instance represents two indistinguishable nodes from p0's perspective
    """

    def __init__(self):
        self.cumulative_regrets = np.zeros(2)  # [pass, bet]
        self.cumulative_strategy = np.zeros(2)  # [pass, bet]
        self.num_actions = 2

    def get_strategy(self, reach_probability):
        # Take positive regrets only
        strategy = np.maximum(self.cumulative_regrets, 0)
        total = np.sum(strategy)
        if total > 0:
            # Normalize to sum to 1
            strategy = strategy / total
        else:
            # Uniform random distribution
            strategy = np.ones(self.num_actions) / self.num_actions

        # Update the cumulative strategy weighted by reach probability
        self.cumulative_strategy += reach_probability * strategy
        return strategy

    def get_average_strategy(self):
        # Normalise cumulative strategy to 1
        total = np.sum(self.cumulative_strategy)
        if total > 0:
            return self.cumulative_strategy / total
        else:
            return np.ones(self.num_actions) / self.num_actions


class Trainer:
    """
    Handles learning strategy and CFR algorithm
    """

    def __init__(self):
        self.game = KuhnPoker()
        self.info_sets = {}

    def get_info_set(self, card, history):
        info_set_key = str(card) + history
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = InformationSet()
        return self.info_sets[info_set_key]

    def cfr(self, cards, history, p0_reach, p1_reach):
        if self.game.is_terminal(history):
            return self.game.get_utility(history, cards[0], cards[1])
        else:
            player = len(history) % 2
            info_set = self.get_info_set(cards[player], history)
            reach_probability = p0_reach if player == 0 else p1_reach
            strategy = info_set.get_strategy(reach_probability)
            if player == 0:
                utility_0_p = self.cfr(
                    cards, history + "p", strategy[0] * reach_probability, p1_reach)
                utility_0_b = self.cfr(
                    cards, history + "b", strategy[1] * reach_probability, p1_reach)
                node_utility = strategy[0] * \
                    utility_0_p + strategy[1] * utility_0_b
                regret_pass = utility_0_p - node_utility
                regret_bet = utility_0_b - node_utility
                info_set.cumulative_regrets[0] += p1_reach * regret_pass
                info_set.cumulative_regrets[1] += p1_reach * regret_bet
            else:
                utility_1_p = self.cfr(
                    cards, history + "p", p0_reach, strategy[0] * reach_probability)
                utility_1_b = self.cfr(
                    cards, history + "b", p0_reach, strategy[1] * reach_probability)
                node_utility = strategy[0] * \
                    utility_1_p + strategy[1] * utility_1_b
                # node util - utility of action
                # node util is what p0 can expect to win given p1's current strategy
                regret_pass = node_utility - utility_1_p
                regret_bet = node_utility - utility_1_b
                info_set.cumulative_regrets[0] += p0_reach * regret_pass
                info_set.cumulative_regrets[1] += p0_reach * regret_bet
            return node_utility

    def train(self, iterations):
        expected_value = 0
        for i in range(iterations):
            # Deal 2 random cards
            cards = random.sample(self.game.cards, 2)

            # Run CFR
            util = self.cfr(cards, "", 1, 1)
            expected_value += util

            if i % 10000 == 0 and i > 0:
                print(f"Iteration {i}:")
                print(f"Expected Value: {expected_value/i}")
        return expected_value / iterations

    def print_strategies(self):
        print("\nOptimal Strategies:")
        for info_set_key in sorted(self.info_sets.keys()):
            info_set = self.info_sets[info_set_key]
            avg_strategy = info_set.get_average_strategy()
            card = info_set_key[0]
            display_card = ["J", "Q", "K"]
            history = info_set_key[1:]

            print(
                f"Card: {display_card[int(card)]}, History: {history}, Pass %: {round(avg_strategy[0], 3)}, Bet %: {round(avg_strategy[1], 3)}")


if __name__ == "__main__":
    trainer = Trainer()
    iterations = 10000000
    expected_value = trainer.train(iterations)
    print(
        f"\nFinal expected value after {iterations} iterations: {expected_value}")
    trainer.print_strategies()
