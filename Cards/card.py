class Card:
    # Define the community region as a class variable
    COMMUNITY_ROI = (270, 690, 1800, 1080)  # (min_x, min_y, max_x, max_y)

    def __init__(self, position, symbol=None, owner=None):
        self.position = position
        self.symbol = symbol
        self.owner = owner

    @staticmethod
    def classify_cards(card):
        """
        Classify cards into community, player1, and player2 based on their positions.

        Args:
            cards (list): List of Card objects.

        Updates:
            Updates the 'owner' attribute for each Card object.
        """
        community_cards = []
        x, y = card.position
        x_min, y_min, x_max, y_max = Card.COMMUNITY_ROI
        if x_min <= x <= x_max and y_min <= y <= y_max:
            card.owner = "community"
            community_cards.append(card)


        # Classify Player 1 cards (2 cards with highest x values)
        elif card.owner != "community" and x > 1980/2:
            card.owner = "player1"

        elif card.owner != "community" and x <= 1980/2:
            card.owner = "player2"

    def __repr__(self):
        """Returns a string representation of the card for debugging and logging."""
        return f"Card(symbol={self.symbol}, position={self.position}, owner={self.owner})"
