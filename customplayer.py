
class CustomPlayer:

    def __init__(self, search_depth=3,
                 iterative=True):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = 0

    def get_move(self, game, legal_moves):

        if not legal_moves:
            return (-1, -1)

        # Did we just start the game? Then, of course, pick the center position.
        if game.move_count == 0:
            return (0, 0)

        # Let's search for a good move!
        best_move_so_far = (-1, -1)

        try:

            if self.iterative == True:
                iterative_search_depth = 1

                while True:
                    best_score_so_far, best_move_so_far = self.alphabeta(game, iterative_search_depth)
                    if best_score_so_far == float("inf") or best_score_so_far == float("-inf"):
                        break
                    iterative_search_depth += 1
                else:
                    raise ValueError('ERR in CustomPlayer.get_move() - invalid param')
            else:
                _, best_move_so_far = self.alphabeta(game, self.search_depth)

        except:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        print(best_move_so_far)
        return best_move_so_far

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):

        # Are there any legal moves left for us to play? If not, then we lost!
        # The maximizing (minimizing) player returns the lowest (highest) possible score.
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            if maximizing_player == True:
                return float("-inf"), (-1, -1)
            else:
                return float("inf"), (-1, -1)

        lowest_score_so_far, highest_score_so_far = float("inf"), float("-inf")
        best_move_so_far = (-1, -1)
        if depth == 1:
            if maximizing_player == True:
                for move in legal_moves:
                    # Evaluate this move.
                    score = self.score(game.forecast_move(move), self)
                    # If this is a score better than beta, no need to search further. Otherwise, remember the best move.
                    if score >= beta:
                        return score, move
                    if score > highest_score_so_far:
                        highest_score_so_far, best_move_so_far = score, move
                print("best" + best_move_so_far)
                print(highest_score_so_far)
                return highest_score_so_far, best_move_so_far
            else:
                for move in legal_moves:
                    # Evaluate this move.
                    score = self.score(game.forecast_move(move), self)
                    # If this is a score worse than alpha, no need to search further. Otherwise, remember the best move.
                    if score <= alpha:
                        return score, move
                    if score < lowest_score_so_far:
                        lowest_score_so_far, best_move_so_far = score, move
                print(lowest_score_so_far)
                print(best_move_so_far)
                return lowest_score_so_far, best_move_so_far

        if maximizing_player == True:
            for move in legal_moves:
                # Evaluate this move in depth.
                score, _ = self.alphabeta(game.forecast_move(move), depth - 1, alpha, beta, maximizing_player=False)
                # If this branch yields a score better than beta, no need to search further.
                if score >= beta:
                    return score, move
                # Otherwise, remember the best move and update alpha.
                if score > highest_score_so_far:
                    highest_score_so_far, best_move_so_far = score, move
                alpha = max(alpha, highest_score_so_far)
            return highest_score_so_far, best_move_so_far
        else:
            for move in legal_moves:
                # Evaluate this move in depth.
                score, _ = self.alphabeta(game.forecast_move(move), depth - 1, alpha, beta, maximizing_player=True)
                # If this branch yields a score worse than alpha, no need to search further.
                if score <= alpha:
                    return score, move
                # Otherwise, remember the best move and update beta.
                if score < lowest_score_so_far:
                    lowest_score_so_far, best_move_so_far = score, move
                beta = min(beta, lowest_score_so_far)
            return lowest_score_so_far, best_move_so_far