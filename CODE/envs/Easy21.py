import numpy as np

#EASY21 ENVIRONMENT
class Easy21:
    def __init__(self): #We initialize from state (0, 0) as we want to consider the first draw a possible state
        self.dealer_card = 0
        self.player_sum  = 0
        self.terminate = False

    def start(self):
        # generates initial state
        self.dealer_card = np.random.randint(1, 11)  # Dealer's first card
        self.player_sum  =  np.random.randint(1, 11)  # Player's initial sum
        self.terminate = False
        return self.dealer_card, self.player_sum

    def step(self, action):
        # Take a step in the environment based on the chosen action (0 for "hit", 1 for "stick")
        if action == 0:  # "hit" action
            card = self.draw_card()
            self.player_sum += card

            if self.player_sum > 21 or self.player_sum < 1:
                self.terminate = True
                self.player_sum -= card # We don't consider these 'out of bound' states
                #print('player went bust')
                return (self.dealer_card, self.player_sum), -1, self.terminate  # Player busts, episode terminates with -1 reward
            
            else:
                return (self.dealer_card, self.player_sum), 0, self.terminate  # Player doesn't bust, episode continues with 0 reward           

        elif action == 1:  # "stick" action
            self.terminate = True

            while self.dealer_card < 17: # Dealer hits until scoring 17
                card = self.draw_card()
                self.dealer_card += card

            if self.dealer_card > 21 or self.dealer_card < 1:
                self.dealer_card -= card # We don't consider these 'out of bound' states
                #print('dealer went bust')
                return (self.dealer_card, self.player_sum), 1, self.terminate  # Dealer busts, episode terminates with +1 reward
            else:
                if self.dealer_card < self.player_sum:
                    return (self.dealer_card, self.player_sum), 1, self.terminate  # Player wins, episode terminates with +1 reward
                elif self.dealer_card > self.player_sum:
                    return (self.dealer_card, self.player_sum), -1, self.terminate  # Dealer wins, episode terminates with -1 reward
                else:
                    return (self.dealer_card, self.player_sum), 0, self.terminate  # Draw, episode terminates with 0 reward
        else:
            raise ValueError("Invalid action: {}. Must be 0 (hit) or 1 (stick).".format(action))

    def draw_card(self):
        # Draws card from deck
        card_color = np.random.choice([-1, 1], p=[1/3, 2/3]) # p=1/3 - red; p=2/3 - black;
        card_value = np.random.randint(1, 11) # from 1 to 10. No aces
        return card_color * card_value