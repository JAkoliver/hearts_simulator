#pragma once

#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <string>

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
    std::array<std::vector<int>, 4> pass_received; // action ids each player was handed

    // Per-round play history: who played which card, and when
    std::array<std::array<bool, 52>, 4> played_by; // [absolute seat][card]
    std::array<int, 52> played_trick;              // trick index card was played on, -1 = unseen

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

    // Start a FRESH MATCH: zero the running totals and restart the pass
    // rotation (left first), then deal. Reset() deliberately preserves
    // totals - it starts the next round of the SAME match.
    void ResetMatch() {
        state.total_scores = {0, 0, 0, 0};
        deal_count = 0;
        Reset();
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

        // Wipe per-round play history
        for (auto& pb : played_by) pb.fill(false);
        played_trick.fill(-1);

        // Passing phase setup: direction cycles Left, Right, Across, Hold
        for (auto& picks : pass_picks) picks.clear();
        for (auto& recv : pass_received) recv.clear();
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
                            pass_received[receiver].push_back(a);
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

        // Record play history: who played this card, and on which trick
        played_by[player][action_id] = true;
        played_trick[action_id] = state.tricks_played;

        // Check for broken hearts
        if (played_card.suit == Suit::Hearts) {
            state.hearts_broken = true;
        }

        // Track voids: playing off-suit while FOLLOWING reveals a void.
        // (A leader reveals nothing — getLedSuit() defaults to Clubs on an
        // empty trick, which used to falsely mark leaders void in clubs.)
        if (!state.current_trick.empty() && played_card.suit != state.getLedSuit() && !state.isFirstTrick()) {
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

    std::array<float, 550> Observe() const {
        return ObserveFor(state.current_player);
    }

    // Observation from an arbitrary seat's perspective (identical layout).
    // Decision-time search uses this to ask the value head about the
    // SEARCHING seat's outlook at a truncated rollout leaf, where the
    // engine's current_player is whoever happens to be next to act.
    std::array<float, 550> ObserveFor(int player) const {
        std::array<float, 550> obs;
        obs.fill(0.0f);

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

        // Block 9: Cards I received in the pass (52 floats: 238-289)
        // The giver knows I hold these until they are played
        for (int a : pass_received[player]) {
            obs[238 + a] = 1.0f;
        }

        // Block 10: Who played what this round, in RELATIVE seats
        // (4 x 52 floats: 290-341 me, 342-393 left(+1), 394-445 across(+2),
        //  446-497 right(+3)). Includes cards in the current trick, which
        //  also attributes the table cards to seats.
        for (int k = 0; k < 4; ++k) {
            int seat = (player + k) % 4;
            for (int c = 0; c < 52; ++c) {
                if (played_by[seat][c]) {
                    obs[290 + k * 52 + c] = 1.0f;
                }
            }
        }

        // Block 11: Play-order timing (52 floats: 498-549)
        // (trick_index + 1) / 13 for played cards, 0 for unseen
        for (int c = 0; c < 52; ++c) {
            if (played_trick[c] >= 0) {
                obs[498 + c] = static_cast<float>(played_trick[c] + 1) / 13.0f;
            }
        }

        return obs;
    }

    // Ground-truth labels for the auxiliary belief head (training only):
    // the actual current hands of the three opponents, in relative seats
    // (0-51 left(+1), 52-103 across(+2), 104-155 right(+3)).
    std::array<float, 156> ObserveOpponentHands() const {
        return ObserveOpponentHandsFor(state.current_player);
    }

    // Same labels from an arbitrary seat's perspective (leaf value records
    // for the oracle head, which needs hands relative to the evaluated seat).
    std::array<float, 156> ObserveOpponentHandsFor(int player) const {
        std::array<float, 156> labels;
        labels.fill(0.0f);
        for (int k = 1; k < 4; ++k) {
            int seat = (player + k) % 4;
            for (const auto& c : state.hands[seat]) {
                labels[(k - 1) * 52 + CardToActionId(c)] = 1.0f;
            }
        }
        return labels;
    }

    const GameState& GetState() const { return state; }

    std::array<int, 4> GetRoundScores() const { return state.round_scores; }
    int GetCurrentPlayer() const { return state.current_player; }
    bool IsPassing() const { return in_passing; }
    int GetPassDirection() const { return pass_direction; }

    // ---- Search support (decision-time determinized search) ----

    HeartsEnv Clone() const { return *this; }

    const std::array<bool, 16>& GetVoidTracker() const { return void_tracker; }
    const std::array<std::array<bool, 52>, 4>& GetPlayedBy() const { return played_by; }
    const std::vector<int>& GetPassPicks(int player) const { return pass_picks[player]; }
    int GetHandSize(int player) const { return static_cast<int>(state.hands[player].size()); }

    // Rewind the passing phase for pass-candidate search: give every player a
    // full 13-card hand (a determinization of the deal) and restart the picks
    // from player 0. Passing is informationally simultaneous, so evaluating a
    // pass from this state is equivalent regardless of real pick order.
    // Only valid while the round is still in (or entering) the passing phase.
    void ResetForPassSearch(const std::array<std::vector<int>, 4>& full_hands) {
        if (pass_direction == 3) {
            throw std::runtime_error("ResetForPassSearch: no passing this round");
        }
        std::array<bool, 52> seen_card;
        seen_card.fill(false);
        for (int i = 0; i < 4; ++i) {
            if (full_hands[i].size() != 13) {
                throw std::runtime_error("ResetForPassSearch: hands must be 13 cards");
            }
            for (int id : full_hands[i]) {
                if (id < 0 || id > 51 || seen_card[id]) {
                    throw std::runtime_error("ResetForPassSearch: invalid or duplicate card");
                }
                seen_card[id] = true;
            }
        }
        for (int i = 0; i < 4; ++i) {
            state.hands[i].clear();
            for (int id : full_hands[i]) {
                state.hands[i].push_back(ActionIdToCard(id));
            }
            SortHand(state.hands[i]);
        }
        for (auto& p : pass_picks) p.clear();
        for (auto& r : pass_received) r.clear();
        in_passing = true;
        state.current_player = 0;
    }

    // Replace all four hands with a determinization (action ids). Validates
    // that the hands form a partition of the correct sizes with no duplicates.
    void SetHands(const std::array<std::vector<int>, 4>& hand_ids) {
        std::array<bool, 52> seen_card;
        seen_card.fill(false);
        for (int i = 0; i < 4; ++i) {
            if (hand_ids[i].size() != state.hands[i].size()) {
                throw std::runtime_error("SetHands: hand size mismatch for player " + std::to_string(i));
            }
            for (int id : hand_ids[i]) {
                if (id < 0 || id > 51 || seen_card[id]) {
                    throw std::runtime_error("SetHands: invalid or duplicate card id " + std::to_string(id));
                }
                seen_card[id] = true;
            }
        }
        for (int i = 0; i < 4; ++i) {
            state.hands[i].clear();
            for (int id : hand_ids[i]) {
                state.hands[i].push_back(ActionIdToCard(id));
            }
            SortHand(state.hands[i]);
        }
    }
};
