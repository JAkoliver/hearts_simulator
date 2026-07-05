#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

using namespace ftxui;

enum class Suit { Clubs, Diamonds, Spades, Hearts };
const std::vector<Suit> ALL_SUITS = {Suit::Clubs, Suit::Diamonds, Suit::Spades, Suit::Hearts};

// 1. New Enum for Macro States
enum class MoonState { NONE, FORTRESS, RUN };

struct Card {
    Suit suit;
    int rank; // 2 to 14

    std::string toString() const {
        std::string s;
        switch (rank) {
            case 11: s += "J"; break;
            case 12: s += "Q"; break;
            case 13: s += "K"; break;
            case 14: s += "A"; break;
            default: s += std::to_string(rank); break;
        }
        switch (suit) {
            case Suit::Clubs: s += "♣"; break;
            case Suit::Diamonds: s += "♦"; break;
            case Suit::Spades: s += "♠"; break;
            case Suit::Hearts: s += "♥"; break;
        }
        return " " + s + " ";
    }
    
    Color getColor() const {
        if (suit == Suit::Diamonds || suit == Suit::Hearts) {
            return Color::RedLight;
        }
        return Color::Default;
    }
};

// Represents a single card played to the current trick
struct PlayedCard {
    int player_id; // 0, 1, 2, or 3
    Card card;
};

struct GameState {
    // ---------------------------------------------------------
    // 1. PLAYER STATE
    // ---------------------------------------------------------
    // We use std::array for the 4 players to keep memory contiguous and fast to copy.
    // player_id 0 is typically the "root" AI making the decision.
    std::array<std::vector<Card>, 4> hands; 
    std::array<int, 4> round_scores = {0, 0, 0, 0}; 
    std::array<int, 4> total_scores = {0, 0, 0, 0}; 

    // ---------------------------------------------------------
    // 2. TRICK STATE
    // ---------------------------------------------------------
    std::vector<PlayedCard> current_trick;
    int current_player = 0;       // Who is next to act (0-3)
    int trick_leader = 0;         // Who led the current trick (0-3)
    int tricks_played = 0;        // 0 to 12. Trick 0 is the restricted first trick.

    // ---------------------------------------------------------
    // 3. GLOBAL GAME RULES STATE
    // ---------------------------------------------------------
    bool hearts_broken = false;   // True if a heart has been sloughed on a previous trick
    
    // Helper to determine the suit that was led (for following suit logic)
    Suit getLedSuit() const {
        if (current_trick.empty()) {
            // Technically shouldn't be called if empty, but safe fallback
            return Suit::Clubs; 
        }
        return current_trick[0].card.suit;
    }
    
    // Helper to check if the current trick is the very first trick of the round
    bool isFirstTrick() const {
        return tricks_played == 0;
    }
};

std::vector<Card> GetLegalMoves(const GameState& state, int player_id) {
    std::vector<Card> legal_moves;
    const std::vector<Card>& hand = state.hands[player_id];
    
    bool is_trick_leader = state.current_trick.empty();
    bool is_first_trick = state.isFirstTrick();

    // ---------------------------------------------------------
    // RULE 1: The Opening Lead of the Game
    // ---------------------------------------------------------
    if (is_first_trick && is_trick_leader) {
        for (const auto& c : hand) {
            if (c.suit == Suit::Clubs && c.rank == 2) {
                legal_moves.push_back(c);
                return legal_moves; // The 2 of Clubs is the ONLY legal move
            }
        }
    }

    // ---------------------------------------------------------
    // RULE 2: Responding to a Lead (Must Follow Suit)
    // ---------------------------------------------------------
    if (!is_trick_leader) {
        Suit led_suit = state.getLedSuit();
        bool can_follow_suit = false;
        
        // Check if we have cards in the led suit
        for (const auto& c : hand) {
            if (c.suit == led_suit) {
                legal_moves.push_back(c);
                can_follow_suit = true;
            }
        }
        
        // If we can follow suit, those are our only legal moves.
        if (can_follow_suit) {
            return legal_moves; 
        }
        
        // If we CANNOT follow suit (we are void), we can slough.
        // However, we must enforce Trick 1 penalty restrictions.
        for (const auto& c : hand) {
            if (is_first_trick) {
                // You cannot play Hearts or the Queen of Spades on Trick 1
                bool is_penalty_card = (c.suit == Suit::Hearts) || (c.suit == Suit::Spades && c.rank == 12);
                if (!is_penalty_card) {
                    legal_moves.push_back(c);
                }
            } else {
                // On any other trick, any slough is legal
                legal_moves.push_back(c); 
            }
        }
        
        // Extreme Edge Case: It is Trick 1, we are void in Clubs, and we hold 
        // LITERALLY NOTHING but penalty cards (13 Hearts/Q♠). 
        // We are legally forced to break the Trick 1 rule.
        if (legal_moves.empty() && is_first_trick) {
            for (const auto& c : hand) {
                legal_moves.push_back(c);
            }
        }
        
        return legal_moves;
    }

    // ---------------------------------------------------------
    // RULE 3: Leading a New Trick (Breaking Hearts)
    // ---------------------------------------------------------
    // If we are here, we are the trick leader for Trick 2 through 13.
    bool holds_only_hearts = true;
    for (const auto& c : hand) {
        if (c.suit != Suit::Hearts) {
            holds_only_hearts = false;
            break;
        }
    }
    
    for (const auto& c : hand) {
        if (c.suit == Suit::Hearts) {
            // Can only lead a Heart if they are broken, OR if we have no other choice
            if (state.hearts_broken || holds_only_hearts) {
                legal_moves.push_back(c);
            }
        } else {
            // Any non-Heart is perfectly legal to lead
            legal_moves.push_back(c);
        }
    }
    
    return legal_moves;
}

void ResolveTrick(GameState& state) {
    // 1. Identify the led suit (the first card played dictates this)
    Suit led_suit = state.current_trick[0].card.suit;

    int highest_rank = -1;
    int trick_winner = -1;
    int trick_points = 0;

    // 2. Single-pass evaluation loop
    for (const auto& played : state.current_trick) {
        
        // Track penalty points: Hearts are 1, Queen of Spades is 13
        if (played.card.suit == Suit::Hearts) {
            state.hearts_broken = true; // Any heart played instantly breaks hearts
            trick_points += 1;
        } else if (played.card.suit == Suit::Spades && played.card.rank == 12) {
            trick_points += 13;
        }

        // Determine the winner: Must match led suit and have the highest rank
        if (played.card.suit == led_suit && played.card.rank > highest_rank) {
            highest_rank = played.card.rank;
            trick_winner = played.player_id;
        }
    }

    // 3. Assign the penalty points to the winner's round score
    state.round_scores[trick_winner] += trick_points;

    // 4. The winner of this trick leads the next trick
    state.trick_leader = trick_winner;
    state.current_player = trick_winner; // They are the first to act next

    // 5. Clean up state for the next trick
    // We use .clear() instead of creating a new vector to avoid reallocating memory
    state.current_trick.clear();
    state.tricks_played++;
}

Card SelectBestMove(const GameState& state, int player_id, const std::vector<Card>& legal_moves) {
    // 1. Forced Move (Only 1 legal option, usually Trick 1 with the 2 of Clubs)
    if (legal_moves.size() == 1) {
        return legal_moves[0];
    }

    bool is_leading = state.current_trick.empty();

    // ---------------------------------------------------------
    // 2. LEADING A TRICK
    // ---------------------------------------------------------
    if (is_leading) {
        Card best_card = legal_moves[0];
        int lowest_rank = 99;
        bool found_safe = false;

        for (const auto& c : legal_moves) {
            bool is_penalty = (c.suit == Suit::Hearts) || (c.suit == Suit::Spades && c.rank == 12);
            // Try to lead the absolute lowest non-penalty card to maintain safety
            if (!is_penalty && c.rank < lowest_rank) {
                lowest_rank = c.rank;
                best_card = c;
                found_safe = true;
            }
        }
        
        // Edge Case: If forced to lead a penalty card (e.g., only Hearts left), lead the lowest one
        if (!found_safe) {
            lowest_rank = 99;
            for (const auto& c : legal_moves) {
                if (c.rank < lowest_rank) {
                    lowest_rank = c.rank;
                    best_card = c;
                }
            }
        }
        return best_card;
    }

    Suit led_suit = state.getLedSuit();
    // GetLegalMoves guarantees that if we CAN follow suit, ALL returned moves match the led suit.
    // Therefore, if the first move matches the led suit, we are following suit.
    bool is_following = (legal_moves[0].suit == led_suit); 

    // ---------------------------------------------------------
    // 3. FOLLOWING SUIT
    // ---------------------------------------------------------
    if (is_following) {
        // Find the current winning rank of the trick
        int current_winner_rank = -1;
        for (const auto& played : state.current_trick) {
            if (played.card.suit == led_suit && played.card.rank > current_winner_rank) {
                current_winner_rank = played.card.rank;
            }
        }

        Card best_duck = {Suit::Clubs, -1};
        Card absolute_lowest = legal_moves[0];
        int lowest_rank = 99;

        for (const auto& c : legal_moves) {
            // Track absolute lowest as a fallback
            if (c.rank < lowest_rank) {
                lowest_rank = c.rank;
                absolute_lowest = c;
            }
            // Track the HIGHEST card that successfully ducks under the current winner
            if (c.rank < current_winner_rank && c.rank > best_duck.rank) {
                best_duck = c;
            }
        }

        // If we can duck, play the highest ducking card to save our lowest umbrellas
        if (best_duck.rank != -1) {
            return best_duck; 
        }
        // If we are forced to step over the winner, play our lowest possible card
        return absolute_lowest; 
    }

    // ---------------------------------------------------------
    // 4. SLOUGHING (Void in the led suit)
    // ---------------------------------------------------------
    // Order of operations: Dump Q♠ -> Dump Highest Heart -> Dump Highest Card
    Card highest_heart = {Suit::Hearts, -1};
    Card highest_card = {Suit::Clubs, -1};

    for (const auto& c : legal_moves) {
        if (c.suit == Suit::Spades && c.rank == 12) {
            return c; // Instant dump of the Q♠
        }
        if (c.suit == Suit::Hearts && c.rank > highest_heart.rank) {
            highest_heart = c;
        }
        if (c.rank > highest_card.rank) {
            highest_card = c;
        }
    }

    if (highest_heart.rank != -1) {
        return highest_heart;
    }
    
    // Slough the highest overall card to shed dangerous liabilities
    return highest_card; 
}

// ---------------------------------------------------------
// PART 1: THE ROLLOUT EXECUTION
// ---------------------------------------------------------
int SimulateRollout(GameState state) {
    // Play until all 13 tricks are resolved
    while (state.tricks_played < 13) {
        int p = state.current_player;
        
        // 1. Get Legal Moves & Select Best
        std::vector<Card> legal_moves = GetLegalMoves(state, p);
        Card chosen_card = SelectBestMove(state, p, legal_moves);
        
        // 2. Remove the chosen card from the player's hand
        auto& hand = state.hands[p];
        hand.erase(std::remove_if(hand.begin(), hand.end(), [&](const Card& c) {
            return c.suit == chosen_card.suit && c.rank == chosen_card.rank;
        }), hand.end());
        
        // 3. Play the card to the table
        state.current_trick.push_back({p, chosen_card});
        
        // 4. Advance to the next player
        state.current_player = (state.current_player + 1) % 4;
        
        // 5. If 4 cards are on the table, resolve the trick
        if (state.current_trick.size() == 4) {
            ResolveTrick(state);
        }
    }
    
    // ---------------------------------------------------------
    // POST-GAME SCORING & SHOOT THE MOON CHECK
    // ---------------------------------------------------------
    for (int i = 0; i < 4; ++i) {
        if (state.round_scores[i] == 26) {
            // Someone shot the moon! 
            if (i == 0) {
                return 0; // Our AI shot the moon. Perfect score!
            } else {
                return 26; // An opponent shot the moon. We eat maximum penalty points.
            }
        }
    }
    
    // If no one shot the moon, return Player 0's (our AI's) penalty points
    return state.round_scores[0];
}


std::vector<Card> dealHand() {
    std::vector<Card> deck;
    for (Suit s : ALL_SUITS) {
        for (int r = 2; r <= 14; ++r) {
            deck.push_back({s, r});
        }
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(deck.begin(), deck.end(), g);

    std::vector<Card> hand;
    int deckIndex = 0;
    // Deal 1 card to the player's hand, discard the next 3 cards. Repeat 13 times.
    while (deckIndex < deck.size() && hand.size() < 13) {
        hand.push_back(deck[deckIndex]);
        deckIndex += 1; // Deal 1
        deckIndex += 3; // Discard 3
    }

    // Sort hand
    std::sort(hand.begin(), hand.end(), [](const Card& a, const Card& b) {
        if (a.suit != b.suit) {
            return static_cast<int>(a.suit) < static_cast<int>(b.suit);
        }
        return a.rank < b.rank;
    });

    return hand;
}

// 2. Updated Macro-Trigger
std::pair<MoonState, Suit> DetectMoonStrategy(const std::vector<Card>& full_hand) {
    // Group cards to check for Unstoppable Run
    std::vector<Card> clubs, diamonds, spades, hearts;
    for (const auto& c : full_hand) {
        if (c.suit == Suit::Clubs) clubs.push_back(c);
        else if (c.suit == Suit::Diamonds) diamonds.push_back(c);
        else if (c.suit == Suit::Spades) spades.push_back(c);
        else if (c.suit == Suit::Hearts) hearts.push_back(c);
    }

    auto isRunSuit = [](const std::vector<Card>& suit_cards) {
        if (suit_cards.size() < 6) return false;
        bool has_a = false, has_k = false, has_q = false;
        for (const auto& c : suit_cards) {
            if (c.rank == 14) has_a = true;
            if (c.rank == 13) has_k = true;
            if (c.rank == 12) has_q = true;
        }
        return has_a && has_k && has_q; // Must be anchored by AKQ
    };

    // Check for lateral entries (Side-suit Aces)
    int total_aces = 0;
    for (const auto& c : full_hand) {
        if (c.rank == 14) total_aces++;
    }

    Suit run_suit = Suit::Clubs; 
    bool found_run = false;
    
    if (isRunSuit(clubs)) { run_suit = Suit::Clubs; found_run = true; }
    else if (isRunSuit(diamonds)) { run_suit = Suit::Diamonds; found_run = true; }
    else if (isRunSuit(spades)) { run_suit = Suit::Spades; found_run = true; }
    else if (isRunSuit(hearts)) { run_suit = Suit::Hearts; found_run = true; }

    // A true Unstoppable Run requires the main suit PLUS at least one lateral entry
    // Since the run suit contains an Ace, total_aces >= 2 guarantees a side-suit Ace.
    if (found_run && total_aces >= 2) {
        return {MoonState::RUN, run_suit};
    }

    // Fallback to High-Card Fortress check
    int hand_weight = 0;
    for (const auto& c : full_hand) {
        if (c.rank == 14) hand_weight += 5;       
        else if (c.rank >= 11) hand_weight += 4;  
        else if (c.rank >= 8) hand_weight += 3;   
        else if (c.rank >= 5) hand_weight += 2;   
        else hand_weight += 1;                    
    }
    if (hand_weight > 45) {
        return {MoonState::FORTRESS, Suit::Clubs}; // Suit is ignored for Fortress
    }

    return {MoonState::NONE, Suit::Clubs};
}

// 3. Rename EvaluateMoonSafety to EvaluateFortressSafety
int EvaluateFortressSafety(const std::vector<Card>& remaining_hand) {
    // Note: A LOWER score is still considered "better" by calculateOptimalPass.
    // Therefore, we apply heavy positive penalties for bad Moon cards (low cards),
    // and heavy negative bonuses for good Moon cards (Face cards).
    int danger = 0; 
    for (const auto& c : remaining_hand) {
        danger -= c.rank * 10; // Baseline reward: higher cards are always slightly better
        if (c.rank <= 7) danger += (8 - c.rank) * 100; 
        if (c.suit == Suit::Clubs && c.rank == 2) danger += 1000; 
        if (c.suit == Suit::Hearts && c.rank < 10) danger += 500; 
        if (c.suit == Suit::Spades && c.rank >= 12) danger -= 300; 
        if (c.suit == Suit::Hearts && c.rank >= 11) danger -= 300; 
    }
    return danger;
}

// 4. New Unstoppable Run Logic
int EvaluateRunSafety(const std::vector<Card>& remaining_hand, Suit run_suit) {
    int danger = 0;
    std::vector<Card> side_suits[4];

    for (const auto& c : remaining_hand) {
        if (c.suit == run_suit) {
            // BONUS: Never pass cards from the long suit. They are guaranteed winners.
            danger -= 500; 
        } else {
            side_suits[static_cast<int>(c.suit)].push_back(c);
            
            // BONUS: Lateral Entries (Side-suit Aces) grant table control
            if (c.rank == 14) {
                danger -= 400; 
            } 
            // PENALTY: Unprotected Face Cards in side suits can accidentally win tricks early
            else if (c.rank >= 11) {
                danger += 300; 
            }
        }
    }
    
    // PENALTY: Failing to engineer artificial voids. 
    // If a side suit is short (1 or 2 cards), it is a severe liability. 
    // We heavily penalize keeping these to force the engine to pass them.
    for (int i = 0; i < 4; ++i) {
        if (static_cast<int>(run_suit) == i) continue;
        size_t size = side_suits[i].size();
        if (size > 0 && size < 3) {
            danger += size * 400; 
        }
    }

    return danger;
}

int EvaluateHandSafety(const std::vector<Card>& remaining_hand) {
    int expected_value_danger = 0;

    // Group cards by suit
    std::vector<Card> clubs, diamonds, spades, hearts;
    for (const auto& c : remaining_hand) {
        if (c.suit == Suit::Clubs) clubs.push_back(c);
        else if (c.suit == Suit::Diamonds) diamonds.push_back(c);
        else if (c.suit == Suit::Spades) spades.push_back(c);
        else if (c.suit == Suit::Hearts) hearts.push_back(c);
    }

    // Sort each suit strictly descending by rank for top-down processing
    auto sortByRankDesc = [](std::vector<Card>& suit_cards) {
        std::sort(suit_cards.begin(), suit_cards.end(), [](const Card& a, const Card& b){
            return a.rank > b.rank; 
        });
    };

    sortByRankDesc(clubs);
    sortByRankDesc(diamonds);
    sortByRankDesc(spades);
    sortByRankDesc(hearts);

    // MICRO-TACTIC 1: Strict Guarding
    // A card is only guarded by cards in the same suit that are of a STRICTLY LOWER rank.
    auto countStrictGuards = [](const std::vector<Card>& suit_cards, int rank) {
        int count = 0;
        for (const auto& c : suit_cards) {
            if (c.rank < rank) count++;
        }
        return count;
    };

    // MICRO-TACTIC 4: Top-Down Suit Safety
    // Sequence of high cards without gaps (from the Ace down) means absolute danger.
    auto evaluateTopDownSafety = [](const std::vector<Card>& suit_cards) {
        int unbroken_danger = 0;
        int expected_top = 14; // Start looking for the Ace
        for (size_t i = 0; i < suit_cards.size(); ++i) {
            if (suit_cards[i].rank == expected_top) {
                // Unbroken top sequence -> absolute trick-winning liability
                unbroken_danger += 200 * (expected_top - 9); 
                expected_top--;
            } else {
                break; // Sequence broken
            }
        }
        return std::max(0, unbroken_danger);
    };

    int liabilities = 0;

    // Process Spades (The Q♠ Threat and Suit Exhaustion)
    int q_spades_danger = 0;
    int a_k_spades_danger = 0;
    for (const auto& c : spades) {
        int guards = countStrictGuards(spades, c.rank);
        if (c.rank == 12) { // Queen of Spades
            int base_q_danger = 1300;
            // Allow this to go negative! If we have 4+ guards, we achieve "Queen Control" 
            // and actively want to hold the card. Cap the bonus at -300.
            q_spades_danger = std::max(-300, base_q_danger - (guards * 400)); 
        } else if (c.rank == 14 || c.rank == 13) { // A♠ or K♠
            if (guards < 3) {
                a_k_spades_danger += 1000 - (guards * 300); // Massive EV penalty spike
            } else {
                a_k_spades_danger += 100; // Well-guarded, acceptable danger
            }
        }
    }
    
    liabilities += q_spades_danger; // No std::max wrapper, allowing the negative bonus to apply
    liabilities += std::max(0, a_k_spades_danger);
    // Removed evaluateTopDownSafety(spades) entirely, as Spades follow different rules

    // MICRO-TACTIC 5: The Spade Umbrella
    // Protect against being passed the A♠, K♠, or Q♠ by holding low spades.
    int low_spades_count = 0;
    for (const auto& c : spades) {
        if (c.rank < 12) low_spades_count++; // Count safe Spades (2 through J)
    }
    
    int incoming_spade_threat = 0;
    // Only worry about the umbrella if we aren't already holding the big threats
    if (q_spades_danger == 0 && a_k_spades_danger == 0) {
        if (low_spades_count == 0) {
            incoming_spade_threat = 400; // Massive risk: Naked to an incoming A/K/Q
        } else if (low_spades_count == 1) {
            incoming_spade_threat = 200; // Still highly vulnerable
        } else if (low_spades_count == 2) {
            incoming_spade_threat = 50;  // Slightly vulnerable
        }
    }
    liabilities += incoming_spade_threat;

    // Process Hearts (Core point liabilities)
    int low_hearts_count = 0; // Track our Heart Umbrella
    
    for (const auto& c : hearts) {
        int guards = countStrictGuards(hearts, c.rank);
        if (c.rank < 10) low_hearts_count++; // Count safe ducking Hearts
        
        if (c.rank >= 10) { 
            int extra_danger = (c.rank - 9) * 80; // 10=80, J=160, Q=240, K=320, A=400
            int guarded_danger = extra_danger - (guards * 50);
            // Danger can be mitigated by guards, but never drops below the card's inherent rank
            liabilities += std::max(c.rank, guarded_danger);
        } else {
            liabilities += c.rank; // Simple rank baseline for safe cards
        }
    }
    liabilities += evaluateTopDownSafety(hearts) * 2; // Absolute danger for sequential top hearts

    // MICRO-TACTIC 6: The Heart Umbrella
    // If we have no low Hearts, we are completely exposed to incoming passes (like a naked A♥).
    // This penalty (+150) mathematically counter-balances the Heart Void bonus (+100), 
    // forcing the AI to keep a safe low Heart unless the rest of the hand is incredibly dangerous.
    if (low_hearts_count == 0) {
        liabilities += 150; 
    }

    // Process Minor Suits (Clubs & Diamonds)
    auto evaluateMinorSuit = [&](const std::vector<Card>& suit_cards, bool is_clubs) {
        int minor_danger = 0;
        int low_minor_count = 0; // Track the minor umbrella
        
        for (const auto& c : suit_cards) {
            int guards = countStrictGuards(suit_cards, c.rank);
            if (c.rank < 10) low_minor_count++;
            
            // Minor suits are only highly dangerous at the top, as they pull the Q♠
            if (c.rank >= 12) { // Q, K, A
                int extra_danger = (c.rank - 10) * 100; // Q=200, K=300, A=400
                int guarded_danger = extra_danger - (guards * 60);
                
                // TRICK 1 EXEMPTION: High clubs are inherently safer than high diamonds 
                // because you cannot take penalty points on the first trick of the game.
                if (is_clubs) {
                    guarded_danger -= 50; 
                }
                
                minor_danger += std::max(c.rank, guarded_danger);
            } else {
                minor_danger += c.rank; // A 10 is inherently more dangerous than a 3
            }
        }
        minor_danger += evaluateTopDownSafety(suit_cards) / 2;
        
        // MICRO-TACTIC 7: Minor Suit Umbrella
        // Add a small penalty (+25) if we have no low cards to duck with.
        // This stops the AI from throwing away a lone 3♦ unless the void is truly necessary.
        if (!suit_cards.empty() && low_minor_count == 0) {
            minor_danger += 25;
        }
        
        return minor_danger;
    };
    
    // Update the calls to the lambda to pass the boolean flag:
    liabilities += evaluateMinorSuit(clubs, true);
    liabilities += evaluateMinorSuit(diamonds, false);

    // MICRO-TACTIC 3: Tactical Voiding Value
    // We only reward voiding if the hand has significant liabilities that 
    // necessitate a sloughing opportunity (voiding a suit to dump points).
    // If the hand is already safe, the bonus is suppressed to discourage 
    // breaking up protective umbrellas.
    
    int total_liabilities = liabilities; // Snapshot current state
    int void_bonus = 0;

    // Tactical Weighting:
    // Heart voids are high value (sloughing penalty points).
    // Minor voids (Diamonds/Clubs) are only valuable if the hand is dangerous.
    if (hearts.empty()) void_bonus += 100 + (total_liabilities / 3);
    
    if (total_liabilities > 300) { // Only reward minor voids if hand is dangerous
        if (diamonds.empty()) void_bonus += 50 + (total_liabilities / 4);
        if (clubs.empty()) void_bonus += 50 + (total_liabilities / 4);
    }
    
    liabilities -= void_bonus;

    int raw_rank_sum = 0;
    for (const auto& c : remaining_hand) {
        raw_rank_sum += c.rank;
    }
    
    // Scale liabilities by 100. We do not cap at 0, allowing negative EV to reward Queen Control.
    expected_value_danger = (liabilities * 100) + raw_rank_sum; 
    return expected_value_danger;
}

struct PassResult {
    std::vector<Card> pass;
    double avg_penalty;
    double moon_shot_prob; // Percentage of games the AI shot the moon
    int total_rollouts;    // How stable is this data?
};

// 5. Update the Router
std::vector<PassResult> calculateOptimalPass(const std::vector<Card>& full_hand) {
    auto strategy = DetectMoonStrategy(full_hand);
    MoonState current_state = strategy.first;
    Suit run_suit = strategy.second;

    std::vector<bool> v(13, false);
    std::fill(v.end() - 3, v.end(), true);

    // ---------------------------------------------------------
    // OFFENSE: PURE HEURISTICS
    // ---------------------------------------------------------
    if (current_state == MoonState::FORTRESS || current_state == MoonState::RUN) {
        struct PassOption {
            double score;
            std::vector<Card> current_pass;
        };
        std::vector<PassOption> all_passes;

        do {
            std::vector<Card> remaining_hand, current_pass;
            for (int i = 0; i < 13; ++i) {
                if (v[i]) current_pass.push_back(full_hand[i]);
                else remaining_hand.push_back(full_hand[i]);
            }
            
            double score = (current_state == MoonState::FORTRESS) 
                ? EvaluateFortressSafety(remaining_hand) 
                : EvaluateRunSafety(remaining_hand, run_suit);

            all_passes.push_back({score, current_pass});
        } while (std::next_permutation(v.begin(), v.end()));

        std::sort(all_passes.begin(), all_passes.end(), [](const PassOption& a, const PassOption& b) {
            return a.score < b.score;
        });

        std::vector<PassResult> results;
        for (int i = 0; i < std::min(3, (int)all_passes.size()); ++i) {
            results.push_back({all_passes[i].current_pass, 0.0, 1.0, 0});
        }
        return results;
    }

    // ---------------------------------------------------------
    // DEFENSE: HEURISTIC PRUNING + CRN MONTE CARLO
    // ---------------------------------------------------------
    
    // Step 1: Grade all 286 passes using the Heuristic Gatekeeper
    struct PassOption {
        int heuristic_score;
        std::vector<Card> current_pass;
        std::vector<Card> remaining_hand;
    };
    std::vector<PassOption> all_passes;

    do {
        std::vector<Card> remaining_hand, current_pass;
        for (int i = 0; i < 13; ++i) {
            if (v[i]) current_pass.push_back(full_hand[i]);
            else remaining_hand.push_back(full_hand[i]);
        }
        int h_score = EvaluateHandSafety(remaining_hand);
        all_passes.push_back({h_score, current_pass, remaining_hand});
    } while (std::next_permutation(v.begin(), v.end()));

    std::sort(all_passes.begin(), all_passes.end(), [](const PassOption& a, const PassOption& b) {
        return a.heuristic_score < b.heuristic_score;
    });

    int top_k = 10;
    if (all_passes.size() > top_k) all_passes.resize(top_k);

    // Step 2: Deduce the 39 unknown cards
    std::vector<Card> unknown_cards;
    for (Suit s : ALL_SUITS) {
        for (int r = 2; r <= 14; ++r) {
            bool in_hand = false;
            for (const auto& c : full_hand) {
                if (c.suit == s && c.rank == r) {
                    in_hand = true;
                    break;
                }
            }
            if (!in_hand) unknown_cards.push_back({s, r});
        }
    }

    std::random_device rd;
    std::mt19937 g(rd());
    
    std::vector<double> mc_total_scores(all_passes.size(), 0.0);
    int rollouts = 0;
    auto start_time = std::chrono::steady_clock::now();

    // Step 3: The Common Random Numbers (CRN) Simulation Loop
    while (true) {
        // Check clock every 10 loops
        if (rollouts % 10 == 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            if (elapsed >= 1500) break; // 1.5 second time budget
        }

        // Shuffle ONCE per universe creation
        std::shuffle(unknown_cards.begin(), unknown_cards.end(), g);

        // Test ALL Top 10 passes against the EXACT SAME universe
        for (size_t i = 0; i < all_passes.size(); ++i) {
            GameState root_state;
            root_state.hands[0] = all_passes[i].remaining_hand; 
            
            // 1. Give Player 0 the EXACT SAME 3 incoming cards for this universe
            root_state.hands[0].push_back(unknown_cards[0]);
            root_state.hands[0].push_back(unknown_cards[1]);
            root_state.hands[0].push_back(unknown_cards[2]);

            // 2. The Hearts Passing Rule: Route all 3 passed cards to ONE specific opponent (Player 1)
            root_state.hands[1].push_back(all_passes[i].current_pass[0]);
            root_state.hands[1].push_back(all_passes[i].current_pass[1]);
            root_state.hands[1].push_back(all_passes[i].current_pass[2]);

            // 3. Distribute the remaining 36 unknown cards to fill the hands
            // Player 1 only needs 10 more cards (they already have the 3 passed cards).
            for (int j = 0; j < 10; ++j) {
                root_state.hands[1].push_back(unknown_cards[3 + j]);
            }
            // Player 2 needs a full 13 cards.
            for (int j = 0; j < 13; ++j) {
                root_state.hands[2].push_back(unknown_cards[13 + j]);
            }
            // Player 3 needs a full 13 cards.
            for (int j = 0; j < 13; ++j) {
                root_state.hands[3].push_back(unknown_cards[26 + j]);
            }

            // Find Trick 1 leader (Player with the 2 of Clubs)
            for (int p = 0; p < 4; ++p) {
                for (const auto& c : root_state.hands[p]) {
                    if (c.suit == Suit::Clubs && c.rank == 2) {
                        root_state.current_player = p;
                        root_state.trick_leader = p;
                        break;
                    }
                }
            }

            mc_total_scores[i] += SimulateRollout(root_state);
        }
        rollouts++;
    }

    // Step 4: Final Evaluation
    std::vector<PassResult> results;
    for (size_t i = 0; i < all_passes.size(); ++i) {
        double avg_mc_score = mc_total_scores[i] / rollouts;
        
        // The flawless tie-breaker
        double adjusted_score = avg_mc_score + (all_passes[i].heuristic_score / 100000.0);
        
        results.push_back({all_passes[i].current_pass, adjusted_score, 0.0, rollouts});
    }

    // Sort by the best score
    std::sort(results.begin(), results.end(), [](const PassResult& a, const PassResult& b) {
        return a.avg_penalty < b.avg_penalty;
    });

    if (results.size() > 3) results.resize(3);
    return results; // Return the whole list to the UI thread
}

int main(int argc, const char* argv[]) {
    auto screen = ScreenInteractive::Fullscreen();

    std::vector<Card> currentHand;
    std::vector<PassResult> recommendedPasses;
    MoonState current_state = MoonState::NONE;

    std::atomic<bool> is_thinking{false};
    std::mutex state_mutex;
    int frame_count = 0;

    auto deal_action = [&]() {
        if (is_thinking) return; // Prevent double-dealing while thinking
        
        is_thinking = true;
        
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            currentHand = dealHand();
            recommendedPasses.clear(); // Clear so UI updates immediately
            current_state = MoonState::NONE; // Temporarily hide moon alerts
        }
        
        // 1. Background thread to run the Monte Carlo AI
        std::thread([&, hand_copy = currentHand]() {
            auto new_pass = calculateOptimalPass(hand_copy);
            auto strat = DetectMoonStrategy(hand_copy);
            
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                recommendedPasses = new_pass;
                current_state = strat.first;
            }
            
            is_thinking = false;
            screen.PostEvent(Event::Custom); // Wake up the UI thread immediately when done
        }).detach();
        
        // 2. Background thread to tick the animation loop
        std::thread([&]() {
            while (is_thinking) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                screen.PostEvent(Event::Custom); // Force the UI to redraw to animate the spinner
            }
        }).detach();
    };

    // Trigger the first deal asynchronously on startup so the UI loads instantly
    deal_action();

    auto button_deal = Button("Reshuffle & Deal (R)", deal_action);

    auto renderer = Renderer(button_deal, [&] {
        frame_count++; // Increment frame for the spinner
        
        std::lock_guard<std::mutex> lock(state_mutex);
        
        Elements card_elements;
        for (const auto& card : currentHand) {
            card_elements.push_back(
                text(card.toString()) | color(card.getColor()) | border
            );
        }

        auto hand_panel = window(text(" Player's Hand "),
            hbox(std::move(card_elements)) | center
        );

        Element pass_panel;
        if (is_thinking) {
            // Render the animated spinner instead of the cards
            pass_panel = window(text(" Recommended Pass "),
                hbox({
                    text("Thinking "),
                    spinner(15, frame_count) | bold | color(Color::Cyan) // Classy Braille dots
                }) | center
            );
        } else if (!recommendedPasses.empty()) {
            // Render the computed cards
            Elements pass_elements;
            for (const auto& card : recommendedPasses[0].pass) {
                pass_elements.push_back(
                    text(card.toString()) | color(card.getColor()) | border
                );
            }

            pass_panel = window(text(" Recommended Pass "),
                hbox(std::move(pass_elements)) | center
            );
        } else {
            pass_panel = window(text(" Recommended Pass "), text("Waiting...") | center);
        }

        Element stats_content;
        if (is_thinking || recommendedPasses.empty()) {
            stats_content = vbox({
                text("Calculating Risk Profiles...") | dim | center
            }) | center;
        } else {
            Elements stat_lines;
            stat_lines.push_back(text("Top 3 Passing Options:") | bold);
            
            for (size_t i = 0; i < recommendedPasses.size(); ++i) {
                std::string pass_str = std::to_string(i + 1) + ". ";
                for (const auto& c : recommendedPasses[i].pass) {
                    pass_str += c.toString() + " ";
                }
                pass_str += "| Risk: " + std::to_string(recommendedPasses[i].avg_penalty);
                stat_lines.push_back(text(pass_str));
            }
            
            stat_lines.push_back(separator());
            
            if (recommendedPasses[0].total_rollouts == 0) {
                stat_lines.push_back(text("Heuristic Override: Moon Shot Locked") | bold | color(Color::RedLight));
                stat_lines.push_back(gauge(1.0) | color(Color::RedLight));
            } else {
                stat_lines.push_back(text("Simulation Health: " + std::to_string(recommendedPasses[0].total_rollouts) + " iterations"));
                float risk_val = std::min(1.0f, std::max(0.0f, (float)(recommendedPasses[0].avg_penalty / 26.0)));
                stat_lines.push_back(gauge(risk_val) | color(Color::Green));
            }
            
            stats_content = vbox(std::move(stat_lines));
        }

        auto stats_panel = window(text(" Statistical Analysis "), stats_content) | flex;
        
        auto instructions = text("Press 'R' to reshuffle. Press 'Q' or 'ESC' to quit.") | dim | center;

        Elements layout_elements;
        layout_elements.push_back(text(" Hearts Card Dealing Simulator ") | bold | center);
        layout_elements.push_back(separator());
        
        if (!is_thinking) { // Only show macro-strategy alerts when we aren't thinking
            if (current_state == MoonState::FORTRESS) {
                layout_elements.push_back(text(" 🚀 HIGH-CARD FORTRESS DETECTED: SHOOTING THE MOON! 🚀 ") | bold | color(Color::RedLight) | center);
                layout_elements.push_back(separator());
            } else if (current_state == MoonState::RUN) {
                layout_elements.push_back(text(" 🔥 UNSTOPPABLE RUN DETECTED: SHOOTING THE MOON! 🔥 ") | bold | color(Color::RedLight) | center);
                layout_elements.push_back(separator());
            }
        }
        
        layout_elements.push_back(hand_panel);
        layout_elements.push_back(pass_panel);
        layout_elements.push_back(stats_panel);
        layout_elements.push_back(hbox({button_deal->Render()}) | center);
        layout_elements.push_back(instructions);

        return vbox(std::move(layout_elements)) | border;
    });

    auto main_component = CatchEvent(renderer, [&](Event event) {
        if (event == Event::Character('r') || event == Event::Character('R')) {
            deal_action();
            return true;
        }
        if (event == Event::Character('q') || event == Event::Character('Q') || event == Event::Escape) {
            screen.Exit();
            return true;
        }
        return false;
    });

    screen.Loop(main_component);

    return 0;
}
