#pragma once

#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <stdexcept>

enum class Suit { Clubs, Diamonds, Spades, Hearts };

struct Card {
    Suit suit;
    int rank; // 2 to 14
};

struct PlayedCard {
    int player_id; // 0, 1, 2, or 3
    Card card;
};

struct GameState {
    std::array<std::vector<Card>, 4> hands; 
    std::array<int, 4> round_scores = {0, 0, 0, 0}; 
    std::array<int, 4> total_scores = {0, 0, 0, 0}; 

    std::vector<PlayedCard> current_trick;
    std::vector<PlayedCard> last_trick; // For UI
    int last_trick_winner = -1;         // For UI
    
    int current_player = 0;       
    int trick_leader = 0;         
    int tricks_played = 0;        

    bool hearts_broken = false;   
    
    Suit getLedSuit() const {
        if (current_trick.empty()) {
            return Suit::Clubs; 
        }
        return current_trick[0].card.suit;
    }
    
    bool isFirstTrick() const {
        return tricks_played == 0;
    }
};

struct StepResult {
    double reward;
    bool done;
};

class HeartsEnv {
private:
    GameState state;
    std::mt19937 rng;
    std::array<int, 52> full_deck;
    std::array<bool, 16> void_tracker;

    // Passing phase (opt-in): direction cycles Left, Right, Across, Hold per deal
    bool passing_enabled;
    int deal_count = 0;
    bool in_passing = false;
    int pass_direction = 3; // 0=left(+1), 1=right(+3), 2=across(+2), 3=hold
    std::array<std::vector<int>, 4> pass_picks; // action ids each player chose to pass

    int CardToActionId(const Card& c) const {
        return (static_cast<int>(c.suit) * 13) + (c.rank - 2);
    }

    Card ActionIdToCard(int id) const {
        Suit s = static_cast<Suit>(id / 13);
        int rank = (id % 13) + 2;
        return {s, rank};
    }

    bool HasSuit(const std::vector<Card>& hand, Suit suit) const {
        for (const auto& c : hand) {
            if (c.suit == suit) return true;
        }
        return false;
    }

    bool HasOnlyHearts(const std::vector<Card>& hand) const {
        for (const auto& c : hand) {
            if (c.suit != Suit::Hearts) return false;
        }
        return true;
    }

    void SortHand(std::vector<Card>& hand) {
        std::sort(hand.begin(), hand.end(), [](const Card& a, const Card& b) {
            if (a.suit != b.suit) return static_cast<int>(a.suit) < static_cast<int>(b.suit);
            return a.rank < b.rank;
        });
    }

    void LocateTwoOfClubs() {
        state.current_player = 0;
        for (int i = 0; i < 4; ++i) {
            for (const auto& c : state.hands[i]) {
                if (c.suit == Suit::Clubs && c.rank == 2) {
                    state.current_player = i;
                    state.trick_leader = i;
                    break;
                }
            }
        }
    }

public:
    HeartsEnv(unsigned int seed = 42, bool enable_passing = false)
        : rng(seed), passing_enabled(enable_passing) {
        for (int i = 0; i < 52; ++i) {
            full_deck[i] = i;
        }
    }

    void Reset() {
        // Preserve total scores across resets
        auto preserved_totals = state.total_scores;
        
        // Wipe state
        state = GameState(); // Reset everything to default
        state.total_scores = preserved_totals; // Restore total scores
        
        void_tracker.fill(false);

        // Shuffle deck
        std::shuffle(full_deck.begin(), full_deck.end(), rng);

        // Deal 13 cards to 4 players
        for (int i = 0; i < 4; ++i) {
            state.hands[i].clear();
            state.hands[i].reserve(13); // Reserve to prevent allocations during Step
            for (int j = 0; j < 13; ++j) {
                state.hands[i].push_back(ActionIdToCard(full_deck[i * 13 + j]));
            }
            // Sort hands to keep things predictable
            SortHand(state.hands[i]);
        }

        // Passing phase setup: direction cycles Left, Right, Across, Hold
        for (auto& picks : pass_picks) picks.clear();
        pass_direction = passing_enabled ? (deal_count % 4) : 3;
        deal_count++;
        in_passing = passing_enabled && pass_direction != 3;

        if (in_passing) {
            // Players 0..3 each select 3 cards to pass (sequential selection is
            // equivalent to simultaneous: nobody observes others' picks)
            state.current_player = 0;
        } else {
            LocateTwoOfClubs();
        }
    }

    StepResult Step(int action_id) {
        StepResult result = {0.0, false};
        Card played_card = ActionIdToCard(action_id);
        int player = state.current_player;

        // --- Passing phase: the action selects a card to pass ---
        if (in_passing) {
            auto& hand = state.hands[player];
            auto it = std::find_if(hand.begin(), hand.end(), [&](const Card& c) {
                return c.suit == played_card.suit && c.rank == played_card.rank;
            });
            if (it != hand.end()) {
                hand.erase(it);
            }
            pass_picks[player].push_back(action_id);

            if (pass_picks[player].size() == 3) {
                if (player == 3) {
                    // All picks made: distribute simultaneously
                    int offset = (pass_direction == 0) ? 1 : (pass_direction == 1) ? 3 : 2;
                    for (int i = 0; i < 4; ++i) {
                        int receiver = (i + offset) % 4;
                        for (int a : pass_picks[i]) {
                            state.hands[receiver].push_back(ActionIdToCard(a));
                        }
                    }
                    for (int i = 0; i < 4; ++i) SortHand(state.hands[i]);
                    in_passing = false;
                    LocateTwoOfClubs();
                } else {
                    state.current_player = player + 1;
                }
            }
            return result;
        }

        // Remove card from hand
        auto& hand = state.hands[player];
        auto it = std::find_if(hand.begin(), hand.end(), [&](const Card& c) {
            return c.suit == played_card.suit && c.rank == played_card.rank;
        });
        if (it != hand.end()) {
            hand.erase(it); // Note: erase on std::vector shifts memory, but does not allocate new heap memory.
        }

        // Check for broken hearts
        if (played_card.suit == Suit::Hearts) {
            state.hearts_broken = true;
        }

        // Track voids
        if (played_card.suit != state.getLedSuit() && !state.isFirstTrick()) {
            void_tracker[player * 4 + static_cast<int>(state.getLedSuit())] = true;
        }

        // Add to trick
        state.current_trick.push_back({player, played_card});

        // If trick is full (4 cards), resolve it
        if (state.current_trick.size() == 4) {
            Suit led_suit = state.getLedSuit();
            int trick_winner = state.current_trick[0].player_id;
            int highest_rank = state.current_trick[0].card.rank;
            int penalty_points = 0;

            for (const auto& pc : state.current_trick) {
                if (pc.card.suit == led_suit && pc.card.rank > highest_rank) {
                    highest_rank = pc.card.rank;
                    trick_winner = pc.player_id;
                }
                if (pc.card.suit == Suit::Hearts) penalty_points += 1;
                if (pc.card.suit == Suit::Spades && pc.card.rank == 12) penalty_points += 13;
            }

            state.round_scores[trick_winner] += penalty_points;
            
            // Assign reward to the trick winner (negative of penalty points)
            if (trick_winner == player) {
                result.reward = -static_cast<double>(penalty_points);
            }

            // Save last trick for UI
            state.last_trick = state.current_trick;
            state.last_trick_winner = trick_winner;

            state.current_player = trick_winner;
            state.trick_leader = trick_winner;
            state.current_trick.clear();
            state.tricks_played++;

            if (state.tricks_played == 13) {
                result.done = true;
                
                // End of Round: Shooting the Moon logic
                bool shot_the_moon = false;
                int moon_shooter = -1;
                for (int i = 0; i < 4; ++i) {
                    if (state.round_scores[i] == 26) {
                        shot_the_moon = true;
                        moon_shooter = i;
                        break;
                    }
                }
                
                if (shot_the_moon) {
                    for (int i = 0; i < 4; ++i) {
                        if (i == moon_shooter) {
                            state.round_scores[i] = 0;
                        } else {
                            state.round_scores[i] = 26;
                        }
                    }
                }
                
                // Accumulate to total scores
                for (int i = 0; i < 4; ++i) {
                    state.total_scores[i] += state.round_scores[i];
                }
            }
        } else {
            // Next player
            state.current_player = (state.current_player + 1) % 4;
        }

        return result;
    }

    std::array<int, 13> GetLegalActions() const {
        std::array<int, 13> legal_actions;
        legal_actions.fill(-1); // -1 Sentinel
        int idx = 0;

        int player = state.current_player;
        const auto& hand = state.hands[player];

        // Passing phase: any remaining card in hand may be passed
        if (in_passing) {
            for (const auto& c : hand) {
                legal_actions[idx++] = CardToActionId(c);
            }
            return legal_actions;
        }

        bool is_leading = state.current_trick.empty();
        bool is_first_trick = state.isFirstTrick();
        Suit led_suit = is_leading ? Suit::Clubs : state.getLedSuit();
        bool has_led_suit = !is_leading && HasSuit(hand, led_suit);

        for (const auto& c : hand) {
            bool legal = true;

            if (is_first_trick) {
                if (is_leading) {
                    if (c.suit != Suit::Clubs || c.rank != 2) legal = false;
                } else {
                    if (has_led_suit) {
                        if (c.suit != led_suit) legal = false;
                    } else {
                        // Cannot play penalty cards on first trick
                        if (c.suit == Suit::Hearts || (c.suit == Suit::Spades && c.rank == 12)) {
                            // Except if you ONLY have penalty cards (edge case)
                            bool has_safe_card = false;
                            for (const auto& hc : hand) {
                                if (hc.suit != Suit::Hearts && !(hc.suit == Suit::Spades && hc.rank == 12)) {
                                    has_safe_card = true;
                                    break;
                                }
                            }
                            if (has_safe_card) legal = false;
                        }
                    }
                }
            } else {
                if (is_leading) {
                    if (c.suit == Suit::Hearts && !state.hearts_broken && !HasOnlyHearts(hand)) {
                        legal = false;
                    }
                } else {
                    if (has_led_suit && c.suit != led_suit) legal = false;
                }
            }

            if (legal) {
                legal_actions[idx++] = CardToActionId(c);
            }
        }

        return legal_actions;
    }

    std::array<float, 238> Observe() const {
        std::array<float, 238> obs;
        obs.fill(0.0f);
        
        int player = state.current_player;
        
        // Block 1: Hand (0-51)
        for (const auto& c : state.hands[player]) {
            obs[CardToActionId(c)] = 1.0f;
        }

        // Block 2: Current Trick (52-103)
        for (const auto& pc : state.current_trick) {
            obs[52 + CardToActionId(pc.card)] = 1.0f;
        }

        // Block 3: History/Previous Tricks (104-155)
        std::array<bool, 52> card_seen;
        card_seen.fill(true); // Assume all cards are in history...

        for (int i = 0; i < 4; ++i) {
            for (const auto& c : state.hands[i]) card_seen[CardToActionId(c)] = false; // ...unless they are in someone's hand
        }
        for (const auto& pc : state.current_trick) {
            card_seen[CardToActionId(pc.card)] = false; // ...or actively on the table
        }
        if (in_passing) {
            // Cards selected for passing are out of hands but NOT played
            for (int i = 0; i < 4; ++i) {
                for (int a : pass_picks[i]) card_seen[a] = false;
            }
        }

        for (int i = 0; i < 52; ++i) {
            if (card_seen[i]) {
                obs[104 + i] = 1.0f;
            }
        }

        // Block 4: Game Context (156-164)
        
        // Scores (4 floats: 156, 157, 158, 159)
        for (int i = 0; i < 4; ++i) {
            obs[156 + i] = static_cast<float>(state.round_scores[i]) / 26.0f;
        }
        
        // Trick Position / Turn (4 floats: 160, 161, 162, 163)
        // One-hot encoding based on how many cards are in the trick
        int trick_pos = state.current_trick.size();
        obs[160 + trick_pos] = 1.0f;
        
        // Hearts Broken (1 float: 164)
        obs[164] = state.hearts_broken ? 1.0f : 0.0f;
        
        // Block 5: Void Tracker (16 floats: 165-180)
        for (int i = 0; i < 16; ++i) {
            obs[165 + i] = void_tracker[i] ? 1.0f : 0.0f;
        }

        // Block 6: Pass Direction one-hot (4 floats: 181-184)
        // 0=left, 1=right, 2=across, 3=hold (also hold when passing disabled)
        obs[181 + pass_direction] = 1.0f;

        // Block 7: In-Passing-Phase flag (1 float: 185)
        obs[185] = in_passing ? 1.0f : 0.0f;

        // Block 8: Cards I passed away this round (52 floats: 186-237)
        // During play these are known cards in the receiving opponent's hand
        for (int a : pass_picks[player]) {
            obs[186 + a] = 1.0f;
        }

        return obs;
    }

    const GameState& GetState() const { return state; }

    std::array<int, 4> GetRoundScores() const { return state.round_scores; }
    int GetCurrentPlayer() const { return state.current_player; }
    bool IsPassing() const { return in_passing; }
    int GetPassDirection() const { return pass_direction; }
};
