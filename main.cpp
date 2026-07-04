#include <algorithm>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

using namespace ftxui;

enum class Suit { Clubs, Diamonds, Spades, Hearts };
const std::vector<Suit> ALL_SUITS = {Suit::Clubs, Suit::Diamonds, Suit::Spades, Suit::Hearts};

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

std::vector<Card> calculateOptimalPass(const std::vector<Card>& full_hand) {
    std::vector<bool> v(13, false);
    std::fill(v.end() - 3, v.end(), true);

    int best_score = std::numeric_limits<int>::max();
    std::vector<Card> best_pass;

    do {
        std::vector<Card> remaining_hand;
        std::vector<Card> current_pass;
        for (int i = 0; i < 13; ++i) {
            if (v[i]) {
                current_pass.push_back(full_hand[i]);
            } else {
                remaining_hand.push_back(full_hand[i]);
            }
        }

        int score = EvaluateHandSafety(remaining_hand);
        if (score < best_score) {
            best_score = score;
            best_pass = current_pass;
        }
    } while (std::next_permutation(v.begin(), v.end()));

    return best_pass;
}

int main(int argc, const char* argv[]) {
    auto screen = ScreenInteractive::Fullscreen();

    std::vector<Card> currentHand = dealHand();
    std::vector<Card> recommendedPass = calculateOptimalPass(currentHand);

    auto deal_action = [&]() {
        currentHand = dealHand();
        recommendedPass = calculateOptimalPass(currentHand);
    };

    auto button_deal = Button("Reshuffle & Deal (R)", deal_action);

    auto renderer = Renderer(button_deal, [&] {
        Elements card_elements;
        for (const auto& card : currentHand) {
            card_elements.push_back(
                text(card.toString()) | color(card.getColor()) | border
            );
        }

        auto hand_panel = window(text(" Player's Hand "),
            hbox(std::move(card_elements)) | center
        );

        Elements pass_elements;
        for (const auto& card : recommendedPass) {
            pass_elements.push_back(
                text(card.toString()) | color(card.getColor()) | border
            );
        }

        auto pass_panel = window(text(" Recommended Pass "),
            hbox(std::move(pass_elements)) | center
        );

        auto stats_panel = window(text(" Statistical Analysis "),
            vbox({
                text("Future odds calculations will be rendered here.") | center,
                text("Placeholder...") | dim | center
            }) | center
        ) | flex;
        
        auto instructions = text("Press 'R' to reshuffle. Press 'Q' or 'ESC' to quit.") | dim | center;

        return vbox({
            text(" Hearts Card Dealing Simulator ") | bold | center,
            separator(),
            hand_panel,
            pass_panel,
            stats_panel,
            hbox({
                button_deal->Render(),
            }) | center,
            instructions
        }) | border;
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
