from collections import defaultdict
from typing import List

class TeamRanker:
    def __init__(self, votes: List[str]):
        self.votes = votes
        self.num_positions = len(votes[0])
        self.team_rankings = defaultdict(lambda: [0] * self.num_positions)

    def build_vote_counts(self):
        for vote in self.votes:
            for pos, team in enumerate(vote):
                self.team_rankings[team][pos] += 1

    def get_sorted_teams(self) -> str:
        self.build_vote_counts()
        #print("team rankings",self.team_rankings)
        teams = list(self.team_rankings.keys())

        def comparator(team):
            return ([-count for count in self.team_rankings[team]], team)

        teams.sort(key=comparator)
        return ''.join(teams)


# === Unit Test ===
def test_team_ranker():
    assert TeamRanker(["ABC", "ACB", "ABC", "ACB", "ACB", "ABC"]).get_sorted_teams() == "ABC"
    assert TeamRanker(["WXYZ", "XYZW"]).get_sorted_teams() == "XWYZ"
    assert TeamRanker(["ZMNAGUEDSJYLBOPHRQICWFXTVK"]).get_sorted_teams() == "ZMNAGUEDSJYLBOPHRQICWFXTVK"

test_team_ranker()
print("All test cases passed.")
