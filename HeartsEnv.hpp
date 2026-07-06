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

public:
    HeartsEnv(unsigned int seed = 42) : rng(seed) {
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
            std::sort(state.hands[i].begin(), state.hands[i].end(), [](const Card& a, const Card& b) {
                if (a.suit != b.suit) return static_cast<int>(a.suit) < static_cast<int>(b.suit);
                return a.rank < b.rank;
            });
        }

        // Skipping passing phase for V1.0. 
        // Locate 2 of Clubs
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

    StepResult Step(int action_id) {
        StepResult result = {0.0, false};
        Card played_card = ActionIdToCard(action_id);
        int player = state.current_player;

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

    std::array<float, 181> Observe() const {
        std::array<float, 181> obs;
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

        return obs;
    }

    const GameState& GetState() const { return state; }

    std::array<int, 4> GetRoundScores() const { return state.round_scores; }
    int GetCurrentPlayer() const { return state.current_player; }
};
