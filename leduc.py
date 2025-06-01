import random
import numpy as np


class LeducPoker:
    """
    This is the Leduc Poker implementation
    """

    def __init__(self):
        # J, Q, K
        self.cards = [0, 0, 1, 1, 2, 2]
        # k = check, b = bet, c = call, r = raise, f = fold
        self.actions = ["check", "bet", "call", "raise", "fold"]

    # Deal cards
    def deal_cards(self):
        cards = random.sample(self.cards, 3)
        # p0, p1, and community card
        return cards[0], cards[1], cards[2]

    # If betting round is complete
    def round_complete(self, round_history):
        """
        A round is complete when:
        - Both players check "kk"
        - One bet, other called "bc"
        - One bet, other raise, call "brc"
        - One bet, other raise, reraise, call, "brrc"
        - Any fold
        """
        if 'f' in round_history:
            return True

        if round_history[-2:] == "kk":
            return True
        if round_history[-2:] == "bc":
            return True
        if round_history[-3:] == "brc":
            return True
        # if round_history[-4:] == "brrc":
        #     return True

        return False

    # If final node is reached
    def is_terminal(self, history):
        if 'f' in history:
            return True

        if '/' in history:
            rounds = history.split('/')
            if len(rounds) == 2 and self.round_complete(rounds[1]):
                return True

        return False

    # Calculate contribution to pot per player
    def calculate_contribution(self, history):
        p0_contrib = 1
        p1_contrib = 1

        if '/' not in history:
            rounds = [history]
        else:
            rounds = history.split('/')

        if len(rounds) >= 1:
            round1 = rounds[0]
            p0_contrib += self.calculate_round_contribution(round1, 0, 2)
            p1_contrib += self.calculate_round_contribution(round1, 1, 2)

        if len(rounds) >= 2:
            round2 = rounds[1]
            p0_contrib += self.calculate_round_contribution(round2, 0, 4)
            p1_contrib += self.calculate_round_contribution(round2, 1, 4)

        return p0_contrib, p1_contrib

    # Calculate contribution by player in a round
    def calculate_round_contribution(self, round_history, player, bet_size):
        current_player = 0
        contributions = [0, 0]

        for action in round_history:
            if action == 'b':  # bet
                contributions[current_player] += bet_size
            elif action == 'r':  # raise
                opponent = 1 - current_player
                # Must call the difference + raise amount
                call_amount = max(
                    0, contributions[opponent] - contributions[current_player])
                contributions[current_player] += call_amount + bet_size
            elif action == 'c':  # call
                opponent = 1 - current_player
                call_amount = contributions[opponent] - \
                    contributions[current_player]
                contributions[current_player] += call_amount
            # 'k' (check) and 'f' (fold) don't add chips
            current_player = 1 - current_player

        return contributions[player]

    # If terminal, get the utility based on final pot and contributions
    def get_utility(self, history, p0_card, p1_card, community_card):
        p0_contrib, p1_contrib = self.calculate_contribution(history)
        total_pot = p0_contrib + p1_contrib

        if 'f' in history:
            last_action_player = (len(history.replace('/', '')) - 1) % 2
            winner = 1 - last_action_player
        else:
            p0_pair = (p0_card == community_card)
            p1_pair = (p1_card == community_card)

            if p0_pair:
                winner = 0
            elif p1_pair:
                winner = 1
            elif p0_card > p1_card:
                winner = 0
            elif p1_card > p0_card:
                winner = 1
            else:  # Tie
                return 0

        if winner == 0:
            return total_pot - p0_contrib
        else:
            return -p0_contrib


class InformationSet:
    """
    Each infoset instance represents two indistinguishable nodes from p0's perspective
    """

    def __init__(self):
        # [check: , bet: , call: , raise: , fold: ]
        self.cumulative_regrets = {}
        # [check: , bet: , call: , raise: , fold: ]
        self.cumulative_strategy = {}

    def get_strategy(self, legal_actions, reach_probability):
        # Take positive regrets only
        regrets = np.array([self.cumulative_regrets.get(action, 0)
                           for action in legal_actions])
        strategy = np.maximum(regrets, 0)
        total = np.sum(strategy)
        if total > 0:
            # Normalize to sum to 1
            strategy = strategy / total
        else:
            # Uniform random distribution
            strategy = np.ones(len(legal_actions)) / len(legal_actions)

        # Update the cumulative strategy weighted by reach probability
        for i, action in enumerate(legal_actions):
            if action not in self.cumulative_strategy:
                self.cumulative_strategy[action] = 0
            self.cumulative_strategy[action] += reach_probability * strategy[i]

        return strategy

    def get_average_strategy(self, legal_actions):
        # Normalise cumulative strategy to 1
        cumulative_strat = np.array(
            [self.cumulative_strategy.get(action, 0) for action in legal_actions])
        total = np.sum(cumulative_strat)
        if total > 0:
            return cumulative_strat / total
        else:
            return np.ones(len(legal_actions)) / len(legal_actions)


class Trainer:
    """
    Handles learning strategy and CFR algorithm
    """

    def __init__(self):
        self.game = LeducPoker()
        self.info_sets = {}

    def get_info_set(self, card, community_card, history):
        if '/' in history:
            info_set_key = f"{card}_{community_card}_{history}"
        else:
            info_set_key = f"{card}_{history}"
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = InformationSet()
        return self.info_sets[info_set_key]

    def get_legal_actions(self, history):
        if '/' in history:
            last_round = history.split('/')[-1]
        else:
            last_round = history

        if last_round == "":
            return ['k', 'b']

        # raises = last_round.count('r')
        if last_round == "k":
            return ['k', 'b']
        elif last_round[-1] == "b":
            return ['c', 'r', 'f']
        elif last_round[-1] == "r":
            return ['c', 'f']
        # elif last_round[-1] == "r" and raises == 2:
        #     return ['c', 'f']
        else:
            raise ValueError(f"Unexpected last action: {last_round[-1]}")

    def cfr(self, cards, history, p0_reach, p1_reach):
        if self.game.is_terminal(history):
            return self.game.get_utility(history, cards[0], cards[1], cards[2])
        else:
            last_round = history.split('/')[-1]
            if self.game.round_complete(last_round):
                history += '/'
            player = len(history.replace('/', '')) % 2
            info_set = self.get_info_set(cards[player], cards[2], history)
            reach_probability = p0_reach if player == 0 else p1_reach
            legal_actions = self.get_legal_actions(history)
            strategy = info_set.get_strategy(legal_actions, reach_probability)
            if player == 0:
                utilities = np.zeros(len(legal_actions))
                i = 0
                for action in legal_actions:
                    utilities[i] = self.cfr(
                        cards, history + action, strategy[i] * reach_probability, p1_reach)
                    i += 1

                node_utility = 0
                for j in range(len(legal_actions)):
                    node_utility += strategy[j] * utilities[j]

                regrets = np.zeros(len(legal_actions))
                for k in range(len(legal_actions)):
                    regrets[k] = utilities[k] - node_utility

                for idx, action in enumerate(legal_actions):
                    if action not in info_set.cumulative_regrets:
                        info_set.cumulative_regrets[action] = 0
                    info_set.cumulative_regrets[action] += p1_reach * \
                        regrets[idx]
            else:
                utilities = np.zeros(len(legal_actions))
                i = 0
                for action in legal_actions:
                    utilities[i] = self.cfr(
                        cards, history + action, p0_reach, strategy[i] * reach_probability)
                    i += 1

                node_utility = 0
                for j in range(len(legal_actions)):
                    node_utility += strategy[j] * utilities[j]

                regrets = np.zeros(len(legal_actions))
                for k in range(len(legal_actions)):
                    regrets[k] = node_utility - utilities[k]

                for idx, action in enumerate(legal_actions):
                    if action not in info_set.cumulative_regrets:
                        info_set.cumulative_regrets[action] = 0
                    info_set.cumulative_regrets[action] += p0_reach * \
                        regrets[idx]
            return node_utility

    def train(self, iterations):
        expected_value = 0
        for i in range(iterations):
            # Deal random cards
            cards = self.game.deal_cards()

            # Run CFR
            util = self.cfr(cards, "", 1, 1)
            expected_value += util

            if i % 50000 == 0 and i > 0:
                print(f"Iteration {i}:")
                print(f"Expected Value: {expected_value/i}")
        return expected_value / iterations

    def print_strategies(self):
        print("\nOptimal Strategies:")
        for info_set_key in sorted(self.info_sets.keys()):
            info_set = self.info_sets[info_set_key]

            # Extract history from info_set_key to get legal actions
            if '_' in info_set_key and '/' in info_set_key:
                # Round 2: format is "card_community_history"
                parts = info_set_key.split('_', 2)
                history = parts[2] if len(parts) > 2 else ""
            else:
                # Round 1: format is "card_history"
                history = info_set_key[2:] if len(info_set_key) > 2 else ""

            legal_actions = self.get_legal_actions(history)
            avg_strategy = info_set.get_average_strategy(legal_actions)

            print(f"Info Set: {info_set_key}")
            for i, action in enumerate(legal_actions):
                print(f"  {action}: {round(avg_strategy[i], 3)}")
            print()

    def save_strategies_to_file(self, filename="leduc_strategies.txt"):
        """Save all information set strategies to a text file"""
        with open(filename, 'w') as f:
            f.write("Leduc Poker Optimal Strategies\n")
            f.write("=" * 50 + "\n\n")

            for info_set_key in sorted(self.info_sets.keys()):
                info_set = self.info_sets[info_set_key]

                # Extract history from info_set_key to get legal actions
                if '_' in info_set_key and '/' in info_set_key:
                    # Round 2: format is "card_community_history"
                    parts = info_set_key.split('_', 2)
                    history = parts[2] if len(parts) > 2 else ""
                else:
                    # Round 1: format is "card_history"
                    history = info_set_key[2:] if len(
                        info_set_key) > 2 else ""

                legal_actions = self.get_legal_actions(history)
                avg_strategy = info_set.get_average_strategy(
                    legal_actions)
                p0_card = info_set_key.split("_")[0]
                if '/' in info_set_key:
                    community_card = info_set_key.split("_")[1]
                    history = info_set_key.split("_")[2]
                else:
                    history = info_set_key.split("_")[1]
                f.write(f"Card: {p0_card}, Comm: {community_card}, history: {history}\n")
                for i, action in enumerate(legal_actions):
                    f.write(
                        f"  {action}: {round(avg_strategy[i], 3)}\n")
                f.write("\n")

            f.write(f"Total Information Sets: {len(self.info_sets)}\n")

        print(f"Strategies saved to {filename}")


if __name__ == "__main__":
    trainer = Trainer()
    iterations = 5000000
    expected_value = trainer.train(iterations)
    print(
        f"\nFinal expected value after {iterations} iterations: {expected_value}")
    trainer.save_strategies_to_file()
