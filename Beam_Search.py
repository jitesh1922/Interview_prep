import heapq
import math

def beam_search_decoder(probabilities, beam_width):
    """
    Beam Search decoder for a sequence of token probabilities.

    Parameters:
    - probabilities: List of list of probabilities for each token at each time step.
                     For example: [[0.1, 0.6, 0.3], [0.3, 0.4, 0.3]]
                     where inner list is the probability distribution over the vocab at that time.
    - beam_width: Number of beams to keep at each step.

    Returns:
    - List of (sequence, score) tuples, with top sequences sorted by score.
    """

    # Start with an empty sequence and score = 0 (log probability)
    sequences = [([], 0)]

    for timestep, token_probs in enumerate(probabilities):
        all_candidates = []

        for seq, score in sequences:
            for token_idx, prob in enumerate(token_probs):
                # Use log(prob) to avoid floating point underflow & work with addition instead of multiplication
                new_score = score + math.log(prob + 1e-10)  # avoid log(0) by adding small value
                new_seq = seq + [token_idx]
                all_candidates.append((new_seq, new_score))

        # Keep top `beam_width` sequences with highest score
        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_width]

    return sequences

# Dummy probabilities for 2 timesteps and vocab size = 3 (tokens: 0, 1, 2)
probs = [
    [0.1, 0.6, 0.3],  # time step 1
    [0.3, 0.4, 0.3]   # time step 2
]

beam_width = 2
result = beam_search_decoder(probs, beam_width)

for seq, score in result:
    print(f"Sequence: {seq}, Score: {score:.4f}")
#output -------->
#Sequence: [1, 1], Score: -1.4271
#Sequence: [1, 0], Score: -1.8971

