from treys import Card as TreysCard, Evaluator

class WinnerDecider:
    def __init__(self):
        self.evaluator = Evaluator()

    @staticmethod
    def convert_to_treys_format(cards):
        """
        Convert the custom Card objects to Treys format.

        Args:
            cards (list): List of Card objects.

        Returns:
            list: List of Treys-formatted card strings.
        """
        treys_card_mapping = {
            "spades": "s",
            "hearts": "h",
            "diamonds": "d",
            "clubs": "c",
            "ace": "A",
            "king": "K",
            "queen": "Q",
            "jack": "J",
            "ten": "T",
            "nine": "9",
            "eight": "8",
            "seven": "7",
            "six": "6",
            "five": "5",
            "four": "4",
            "three": "3",
            "two": "2"
        }

        treys_cards = []
        for card in cards:
            rank, suit = card.symbol.split(" of ")
            treys_card = f"{treys_card_mapping[rank]}{treys_card_mapping[suit]}"
            treys_cards.append(TreysCard.new(treys_card))

        return treys_cards

    def decide_winner(self, player1_cards, player2_cards, community_cards):
        """
        Decide the winner between Player 1 and Player 2 using the Treys library.

        Args:
            player1_cards (list): List of Card objects for Player 1.
            player2_cards (list): List of Card objects for Player 2.
            community_cards (list): List of Card objects for the community.

        Returns:
            str: "player1", "player2", or "draw".
        """
        # Convert all cards to Treys format
        player1_hand = self.convert_to_treys_format(player1_cards)
        player2_hand = self.convert_to_treys_format(player2_cards)
        community = self.convert_to_treys_format(community_cards)

        # Evaluate hands
        player1_score = self.evaluator.evaluate(community, player1_hand)
        player2_score = self.evaluator.evaluate(community, player2_hand)

        # Determine the winner
        if player1_score < player2_score:
            return "player1"
        elif player2_score < player1_score:
            return "player2"
        else:
            return "draw"
